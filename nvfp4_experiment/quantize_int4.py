"""
Phase 2: INT4 Post-Training Quantization (PTQ) of the BF16 checkpoint.

Applies per-group symmetric INT4 weight quantization (group_size=128) to every
nn.Linear layer except lm_head (weight-tied to the embedding).

Measures:
  - Validation loss / perplexity (accuracy degradation vs BF16)
  - Token-generation throughput (tok/sec)
  - Quantized model memory footprint (MB)

Results written to results.json under key "int4".
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

GROUP_SIZE = 128   # standard for GPTQ / AWQ-style INT4


# ── INT4 Linear module ────────────────────────────────────────────────────────

class Int4Linear(nn.Module):
    """
    Per-group symmetric INT4 weight quantization.

    Weights are quantized to the range [-8, 7] (4-bit signed integer).
    Each group of GROUP_SIZE weights shares one float32 scale factor.
    Dequantisation is performed on-the-fly at inference time in the input dtype.

    Memory layout (approximate):
      Original BF16: out × in × 2 bytes
      INT4 (int8 storage, no packing): out × in × 1 byte
      Scales: out × (in / group_size) × 4 bytes
      Net reduction ≈ 1.8-2x over BF16 with int8 storage, would be ~3.8x with
      proper nibble packing (omitted here to keep code readable).
    """

    def __init__(self, weight: torch.Tensor, bias, group_size: int = GROUP_SIZE):
        super().__init__()
        out_f, in_f = weight.shape
        self.in_features  = in_f
        self.out_features = out_f
        self.group_size   = group_size

        gs  = min(group_size, in_f)
        pad = (gs - in_f % gs) % gs

        w = weight.detach().float()
        if pad > 0:
            w = F.pad(w, (0, pad))           # right-pad last dim

        n_groups = w.shape[1] // gs
        w_g = w.reshape(out_f, n_groups, gs)  # (out, G, gs)

        # Symmetric: scale = max(|w|) / 7  → maps to [-8..7]
        scales = w_g.abs().amax(dim=-1).clamp(min=1e-8) / 7.0   # (out, G)
        w_int4 = (w_g / scales.unsqueeze(-1)).round().clamp(-8, 7).to(torch.int8)

        self.register_buffer('w_int4', w_int4)          # (out, G, gs) int8
        self.register_buffer('scales', scales.float())  # (out, G)     fp32
        self._in_f_orig = in_f

        self.bias = nn.Parameter(bias.detach().clone()) if bias is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_f, n_groups, gs = self.w_int4.shape
        # Dequantise → (out, in_f_padded) → trim padding → cast to input dtype
        w = (self.w_int4.float() * self.scales.unsqueeze(-1)).reshape(out_f, -1)
        w = w[:, :self._in_f_orig].to(x.dtype)
        return F.linear(x, w, self.bias)

    def extra_repr(self):
        return (f'in={self.in_features}, out={self.out_features}, '
                f'group_size={self.group_size}')


# ── Model quantisation helper ─────────────────────────────────────────────────

def _quantize_recursive(module: nn.Module, skip_names: set, group_size: int):
    """Recursively replace nn.Linear children (not in skip_names) with Int4Linear."""
    for name, child in list(module.named_children()):
        if name in skip_names:
            continue
        if isinstance(child, nn.Linear):
            setattr(module, name, Int4Linear(child.weight, child.bias, group_size))
        else:
            _quantize_recursive(child, skip_names, group_size)


def quantize_model_int4(model: GPT, group_size: int = GROUP_SIZE) -> GPT:
    # lm_head is weight-tied to the token embedding — leave it in BF16
    _quantize_recursive(model, skip_names={'lm_head'}, group_size=group_size)
    return model


def model_memory_mb(model: nn.Module) -> float:
    total = sum(p.untyped_storage().nbytes() for p in
                {*model.parameters(), *model.buffers()})
    return total / 1024**2


# ── Data loader ───────────────────────────────────────────────────────────────

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
    eval_iters = 100
    gen_tokens = 500
    batch_size = 32    # smaller batch for inference measurement

    # ── Load BF16 checkpoint ───────────────────────────────────────────────────
    ckpt_path = os.path.join(OUT_DIR, 'ckpt_fp16.pt')
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    args       = ckpt['model_args']
    block_size = args['block_size']
    cfg        = GPTConfig(**args)
    model      = GPT(cfg)

    sd = ckpt['model']
    # Strip torch.compile prefix if present
    sd = {k[len('_orig_mod.'):] if k.startswith('_orig_mod.') else k: v
          for k, v in sd.items()}
    model.load_state_dict(sd)

    bf16_size_mb = model_memory_mb(model)
    print(f"BF16 model in-memory: {bf16_size_mb:.1f} MB")

    # ── Apply INT4 PTQ ─────────────────────────────────────────────────────────
    print(f"Quantising to INT4 (group_size={GROUP_SIZE})…")
    model = quantize_model_int4(model, GROUP_SIZE)
    int4_size_mb = model_memory_mb(model)
    print(f"INT4 model in-memory: {int4_size_mb:.1f} MB  "
          f"({bf16_size_mb/int4_size_mb:.2f}x compression vs BF16)")

    model = model.to(device).eval()
    torch.cuda.reset_peak_memory_stats()

    # ── Evaluate val loss ──────────────────────────────────────────────────────
    print("Evaluating val loss…")
    total_loss = 0.0
    with torch.no_grad():
        for _ in range(eval_iters):
            X, Y = get_batch('val', block_size, batch_size)
            _, loss = model(X, Y)
            total_loss += loss.item()
    val_loss = total_loss / eval_iters

    # ── Measure token-generation throughput ───────────────────────────────────
    print("Measuring inference throughput…")
    start_ids = torch.zeros((1, 1), dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(3):                              # warm-up
            model.generate(start_ids, max_new_tokens=50)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        model.generate(start_ids, max_new_tokens=gen_tokens)
    torch.cuda.synchronize()
    toks_per_sec = gen_tokens / (time.perf_counter() - t0)

    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024**2

    print(f"\n{'='*60}")
    print(f"INT4 PTQ Results")
    print(f"  Val loss           : {val_loss:.4f}  (PPL {math.exp(val_loss):.2f})")
    print(f"  INT4 model size    : {int4_size_mb:.1f} MB")
    print(f"  Compression        : {bf16_size_mb/int4_size_mb:.2f}x vs BF16")
    print(f"  Inference speed    : {toks_per_sec:.1f} tok/sec")
    print(f"  Peak VRAM          : {peak_vram_mb:.1f} MB")
    print(f"{'='*60}\n")

    # ── Persist results ────────────────────────────────────────────────────────
    results_path = os.path.join(OUT_DIR, 'results_fp.json')
    results = {}
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
    results['int4'] = {
        'val_loss':                round(val_loss, 4),
        'perplexity':              round(math.exp(val_loss), 2),
        'model_size_mb':           round(int4_size_mb, 1),
        'compression_vs_bf16':     round(bf16_size_mb / int4_size_mb, 2),
        'inference_toks_per_sec':  round(toks_per_sec, 1),
        'peak_vram_mb':            round(peak_vram_mb, 1),
    }
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results → {results_path}\n")


if __name__ == '__main__':
    main()
