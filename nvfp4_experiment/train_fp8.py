"""
Phase 3: Native FP8 Training via Transformer Engine (separate from NVFP4 code).

Uses te.Linear layers (E4M3 FP8, DelayedScaling recipe) inside the Attention
and MLP blocks.  On this RTX 5070 Ti Laptop (sm_120 / Blackwell), DelayedScaling
is the operative FP8 path — it caches per-tensor amax values from the previous
step and avoids the shared-memory bottleneck that blocks NVFP4BlockScaling in WSL2.

Design choices that differ from train_fp16.py:
  - te.Linear replaces nn.Linear in transformer blocks (see model_te.py).
  - Master weights stored as BF16 (TE's recommended precision for stability;
    the FP8 quantisation happens entirely inside fp8_autocast).
  - No GradScaler needed: TE handles its own loss/grad scaling internally.

Architecture and hyper-parameters are identical to train_fp16.py for a fair
apples-to-apples comparison.

Checkpoint saved as: ckpt_fp8.pt
Results written to results_fp.json under key "fp8_native".
"""

import os, sys, time, math, pickle, json
import numpy as np
import torch
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
NANOGPT_DIR = os.path.join(SCRIPT_DIR, '..', 'nanoGPT')
DATA_DIR    = os.path.join(NANOGPT_DIR, 'data', 'shakespeare_char')
OUT_DIR     = SCRIPT_DIR
sys.path.insert(0, SCRIPT_DIR)

from model_te import GPTConfig, GPTTE

# ── Identical model config to train_fp16.py ───────────────────────────────────
N_LAYER    = 12
N_HEAD     = 12
N_EMBD     = 768
BLOCK_SIZE = 256   # divisible by 16 (FP8 tensor-core alignment)
BIAS       = False
DROPOUT    = 0.0

BATCH_SIZE     = 32
MAX_ITERS      = 5000
EVAL_ITERS     = 100
EVAL_INTERVAL  = 500
LOG_INTERVAL   = 50

LR             = 6e-4
WEIGHT_DECAY   = 1e-1
BETA1, BETA2   = 0.9, 0.95
GRAD_CLIP      = 1.0
WARMUP_ITERS   = 200
LR_DECAY_ITERS = 5000
MIN_LR         = 6e-5

DEVICE = 'cuda'
SEED   = 1337
# ─────────────────────────────────────────────────────────────────────────────


def cosine_lr(it):
    if it < WARMUP_ITERS:
        return LR * (it + 1) / (WARMUP_ITERS + 1)
    if it > LR_DECAY_ITERS:
        return MIN_LR
    ratio = (it - WARMUP_ITERS) / (LR_DECAY_ITERS - WARMUP_ITERS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return MIN_LR + coeff * (LR - MIN_LR)


def get_batch(split):
    data = np.memmap(
        os.path.join(DATA_DIR, 'train.bin' if split == 'train' else 'val.bin'),
        dtype=np.uint16, mode='r'
    )
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([torch.from_numpy(data[i:i+BLOCK_SIZE].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+BLOCK_SIZE].astype(np.int64)) for i in ix])
    return (x.pin_memory().to(DEVICE, non_blocking=True),
            y.pin_memory().to(DEVICE, non_blocking=True))


@torch.no_grad()
def estimate_loss(model, recipe):
    model.eval()
    out = {}
    for split in ('train', 'val'):
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    with open(os.path.join(DATA_DIR, 'meta.pkl'), 'rb') as f:
        vocab_size = pickle.load(f)['vocab_size']
    print(f"Vocab size: {vocab_size}")

    cfg   = GPTConfig(n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD,
                      block_size=BLOCK_SIZE, bias=BIAS,
                      vocab_size=vocab_size, dropout=DROPOUT)
    model = GPTTE(cfg).to(DEVICE)

    # BF16 master weights — TE quantises to FP8 (E4M3) inside fp8_autocast
    model = model.to(torch.bfloat16)
    print(f"Parameters: {model.get_num_params()/1e6:.2f}M  (master weights: bfloat16)")
    print(f"FP8 recipe: DelayedScaling (E4M3) — Blackwell sm_120 Tensor Cores")

    recipe    = DelayedScaling()
    optimizer = model.configure_optimizers(WEIGHT_DECAY, LR, (BETA1, BETA2), 'cuda')

    torch.cuda.reset_peak_memory_stats()
    X, Y = get_batch('train')
    t_start = time.perf_counter()
    t0      = t_start
    iter_times     = []
    final_val_loss = float('inf')

    for it in range(MAX_ITERS + 1):
        lr = cosine_lr(it)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        if it % EVAL_INTERVAL == 0:
            losses = estimate_loss(model, recipe)
            print(f"step {it:5d}: train {losses['train']:.4f}  val {losses['val']:.4f}")
            final_val_loss = losses['val']

        if it == MAX_ITERS:
            break

        # ── FP8 forward ───────────────────────────────────────────────────────
        with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
            _, loss = model(X, Y)
        # ─────────────────────────────────────────────────────────────────────

        X, Y = get_batch('train')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        t1 = time.perf_counter()
        if it >= 20:
            iter_times.append(t1 - t0)
        t0 = t1

        if it % LOG_INTERVAL == 0:
            print(f"  iter {it:5d}: loss {loss.item():.4f}  lr {lr:.2e}")

    total_time    = time.perf_counter() - t_start
    iters_per_sec = 1.0 / (sum(iter_times) / len(iter_times)) if iter_times else 0.0
    peak_vram_mb  = torch.cuda.max_memory_allocated() / 1024**2

    print(f"\n{'='*60}")
    print(f"FP8 Native Training Complete")
    print(f"  Final val loss : {final_val_loss:.4f}  (PPL {math.exp(final_val_loss):.2f})")
    print(f"  Total time     : {total_time:.1f} s")
    print(f"  Iters/sec      : {iters_per_sec:.2f}")
    print(f"  Peak VRAM      : {peak_vram_mb:.1f} MB")
    print(f"{'='*60}\n")

    ckpt_path = os.path.join(OUT_DIR, 'ckpt_fp8.pt')
    torch.save({
        'model':      model.state_dict(),
        'model_args': dict(n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD,
                           block_size=BLOCK_SIZE, bias=BIAS,
                           vocab_size=vocab_size, dropout=DROPOUT),
        'iter_num':   MAX_ITERS,
        'val_loss':   final_val_loss,
    }, ckpt_path)
    print(f"Checkpoint → {ckpt_path}")

    results_path = os.path.join(OUT_DIR, 'results_fp.json')
    results = {}
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
    results['fp8_native'] = {
        'precision':          'FP8 E4M3 DelayedScaling (Blackwell Tensor Cores)',
        'val_loss':           round(final_val_loss, 4),
        'perplexity':         round(math.exp(final_val_loss), 2),
        'total_train_time_s': round(total_time, 1),
        'iters_per_sec':      round(iters_per_sec, 2),
        'peak_vram_mb':       round(peak_vram_mb, 1),
    }
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results   → {results_path}\n")


if __name__ == '__main__':
    main()
