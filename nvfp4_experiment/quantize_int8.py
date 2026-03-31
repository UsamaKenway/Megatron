"""
Phase 2: INT8 Post-Training Quantization of the FP16 checkpoint.

Applies per-channel symmetric INT8 weight quantization to every nn.Linear
layer except lm_head (weight-tied to the embedding).

  scale[out_channel] = max(|w[out_channel, :]|) / 127
  w_int8 = round(w / scale).clamp(-128, 127)

This is the standard 8-bit weight quantisation used by llama.cpp / GPTQ-INT8.
Activations remain in float16 at inference time (weight-only INT8).

Measures:
  - Validation loss / perplexity  (accuracy degradation vs FP16)
  - Token-generation throughput   (tok/sec)
  - Quantised model memory footprint (MB)

Results written to results_fp.json under key "int8".
"""

import os, sys, time, math, pickle, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
NANOGPT_DIR = os.path.join(SCRIPT_DIR, '..', 'nanoGPT')
DATA_DIR    = os.path.join(NANOGPT_DIR, 'data', 'shakespeare_char')
OUT_DIR     = SCRIPT_DIR
sys.path.insert(0, NANOGPT_DIR)

from model import GPTConfig, GPT

# ── INT8 Linear ───────────────────────────────────────────────────────────────

class Int8Linear(nn.Module):
    """
    Per-channel symmetric INT8 weight quantisation.

    Weight stored as int8 (1 byte/element vs 2 bytes for fp16) → ~2x compression.
    One fp32 scale per output channel.  Dequantisation is on-the-fly.

    This corresponds to weight-only INT8 as used in llama.cpp W8A16.
    """

    def __init__(self, weight: torch.Tensor, bias):
        super().__init__()
        out_f, in_f = weight.shape
        self.in_features  = in_f
        self.out_features = out_f

        w = weight.detach().float()
        # Per-channel scale: one scale per output row
        scales  = w.abs().amax(dim=1).clamp(min=1e-8) / 127.0   # (out_f,)
        w_int8  = (w / scales.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)

        self.register_buffer('w_int8',  w_int8)          # (out, in)  int8
        self.register_buffer('scales',  scales.float())  # (out,)     fp32

        self.bias = nn.Parameter(bias.detach().clone()) if bias is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantise per-channel: w_fp = w_int8 * scales[:, None]
        w = (self.w_int8.float() * self.scales.unsqueeze(1)).to(x.dtype)
        return F.linear(x, w, self.bias)

    def extra_repr(self):
        return f'in={self.in_features}, out={self.out_features}'


# ── Quantise model ────────────────────────────────────────────────────────────

def _quantize_recursive(module: nn.Module, skip_names: set):
    for name, child in list(module.named_children()):
        if name in skip_names:
            continue
        if isinstance(child, nn.Linear):
            setattr(module, name, Int8Linear(child.weight, child.bias))
        else:
            _quantize_recursive(child, skip_names)


def quantize_model_int8(model: GPT) -> GPT:
    _quantize_recursive(model, skip_names={'lm_head'})
    return model


def model_memory_mb(model: nn.Module) -> float:
    total = sum(p.untyped_storage().nbytes() for p in
                {*model.parameters(), *model.buffers()})
    return total / 1024**2


# ── Data ──────────────────────────────────────────────────────────────────────

def get_batch(split, block_size, batch_size):
    data = np.memmap(
        os.path.join(DATA_DIR, 'train.bin' if split == 'train' else 'val.bin'),
        dtype=np.uint16, mode='r'
    )
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
    return (x.pin_memory().to('cuda', non_blocking=True),
            y.pin_memory().to('cuda', non_blocking=True))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device     = 'cuda'
    eval_iters = 200
    gen_tokens = 500
    batch_size = 16   # smaller batch for inference measurement

    ckpt_path = os.path.join(OUT_DIR, 'ckpt_fp16.pt')
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    args       = ckpt['model_args']
    block_size = args['block_size']
    cfg        = GPTConfig(**args)
    model      = GPT(cfg)

    sd = {k[len('_orig_mod.'):] if k.startswith('_orig_mod.') else k: v
          for k, v in ckpt['model'].items()}
    model.load_state_dict(sd)

    fp16_size_mb = model_memory_mb(model)
    print(f"FP16 model in-memory: {fp16_size_mb:.1f} MB")

    print("Quantising to INT8 (per-channel symmetric)…")
    model = quantize_model_int8(model)
    int8_size_mb = model_memory_mb(model)
    print(f"INT8 model in-memory: {int8_size_mb:.1f} MB  "
          f"({fp16_size_mb/int8_size_mb:.2f}× compression vs FP16)")

    model = model.to(device).half().eval()   # activations in fp16
    torch.cuda.reset_peak_memory_stats()

    # ── Val loss ───────────────────────────────────────────────────────────────
    print("Evaluating val loss…")
    ctx = torch.amp.autocast(device_type='cuda', dtype=torch.float16)
    total_loss = 0.0
    with torch.no_grad():
        for _ in range(eval_iters):
            X, Y = get_batch('val', block_size, batch_size)
            with ctx:
                _, loss = model(X, Y)
            total_loss += loss.item()
    val_loss = total_loss / eval_iters

    # ── Inference throughput ───────────────────────────────────────────────────
    print("Measuring inference throughput…")
    start_ids = torch.zeros((1, 1), dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(3):
            model.generate(start_ids, max_new_tokens=50)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        model.generate(start_ids, max_new_tokens=gen_tokens)
    torch.cuda.synchronize()
    toks_per_sec = gen_tokens / (time.perf_counter() - t0)

    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024**2

    print(f"\n{'='*60}")
    print(f"INT8 PTQ Results")
    print(f"  Val loss         : {val_loss:.4f}  (PPL {math.exp(val_loss):.2f})")
    print(f"  INT8 model size  : {int8_size_mb:.1f} MB")
    print(f"  Compression      : {fp16_size_mb/int8_size_mb:.2f}× vs FP16")
    print(f"  Inference speed  : {toks_per_sec:.1f} tok/sec")
    print(f"  Peak VRAM        : {peak_vram_mb:.1f} MB")
    print(f"{'='*60}\n")

    results_path = os.path.join(OUT_DIR, 'results_fp.json')
    results = {}
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
    results['int8'] = {
        'val_loss':               round(val_loss, 4),
        'perplexity':             round(math.exp(val_loss), 2),
        'model_size_mb':          round(int8_size_mb, 1),
        'compression_vs_fp16':    round(fp16_size_mb / int8_size_mb, 2),
        'inference_toks_per_sec': round(toks_per_sec, 1),
        'peak_vram_mb':           round(peak_vram_mb, 1),
    }
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results → {results_path}\n")


if __name__ == '__main__':
    main()
