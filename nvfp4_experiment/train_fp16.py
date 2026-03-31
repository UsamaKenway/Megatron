"""
Phase 1: FP16 Baseline Training on shakespeare_char.

Architecture: 12-layer · 12-head · 768-dim GPT (GPT-2 scale, ~85M params on
  this char vocab of 65; same weight-matrix sizes as the 124M GPT-2).

float16 + GradScaler (standard reduced-precision training).
Records: peak VRAM, iter/sec, total time, final val loss → results_fp.json.
Checkpoint saved as: ckpt_fp16.pt
"""

import os, sys, time, math, pickle, json
from contextlib import nullcontext
import numpy as np
import torch
import torch.nn as nn

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
NANOGPT_DIR = os.path.join(SCRIPT_DIR, '..', 'nanoGPT')
DATA_DIR    = os.path.join(NANOGPT_DIR, 'data', 'shakespeare_char')
OUT_DIR     = SCRIPT_DIR
sys.path.insert(0, NANOGPT_DIR)

from model import GPTConfig, GPT

# ── Model config (GPT-2 scale, shared across all FP phases) ───────────────────
N_LAYER    = 12
N_HEAD     = 12
N_EMBD     = 768
BLOCK_SIZE = 256   # divisible by 16 (FP8 tensor-core alignment)
BIAS       = False
DROPOUT    = 0.0

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE     = 32     # divisible by 16
MAX_ITERS      = 5000
EVAL_ITERS     = 100
EVAL_INTERVAL  = 500
LOG_INTERVAL   = 50

LR             = 6e-4   # standard GPT-2 LR
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
def estimate_loss(model, ctx):
    model.eval()
    out = {}
    for split in ('train', 'val'):
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            with ctx:
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
    model = GPT(cfg).to(DEVICE)
    print(f"Parameters: {model.get_num_params()/1e6:.2f}M  (dtype: float32 → cast to float16 in autocast)")

    # float16 autocast + GradScaler (prevents underflow in backward pass)
    ctx    = torch.amp.autocast(device_type='cuda', dtype=torch.float16)
    scaler = torch.amp.GradScaler('cuda', enabled=True)

    optimizer = model.configure_optimizers(WEIGHT_DECAY, LR, (BETA1, BETA2), 'cuda')

    torch.cuda.reset_peak_memory_stats()
    X, Y = get_batch('train')
    t_start = time.perf_counter()
    t0 = t_start
    iter_times     = []
    final_val_loss = float('inf')

    for it in range(MAX_ITERS + 1):
        lr = cosine_lr(it)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        if it % EVAL_INTERVAL == 0:
            losses = estimate_loss(model, ctx)
            print(f"step {it:5d}: train {losses['train']:.4f}  val {losses['val']:.4f}")
            final_val_loss = losses['val']

        if it == MAX_ITERS:
            break

        with ctx:
            _, loss = model(X, Y)
        X, Y = get_batch('train')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
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
    print(f"FP16 Training Complete")
    print(f"  Final val loss : {final_val_loss:.4f}  (PPL {math.exp(final_val_loss):.2f})")
    print(f"  Total time     : {total_time:.1f} s")
    print(f"  Iters/sec      : {iters_per_sec:.2f}")
    print(f"  Peak VRAM      : {peak_vram_mb:.1f} MB")
    print(f"{'='*60}\n")

    ckpt_path = os.path.join(OUT_DIR, 'ckpt_fp16.pt')
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
    results['fp16'] = {
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
