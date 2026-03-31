"""
Phase 1: BF16 Baseline Training on shakespeare_char.

Trains a 6-layer GPT (384-dim, ~10M params) with bfloat16 autocast.
Records: peak VRAM, iter/sec, total time, final val loss → results.json.
Checkpoint saved as: ckpt_bf16.pt
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

# ── Model (identical across all three phases) ─────────────────────────────────
N_LAYER    = 6
N_HEAD     = 6
N_EMBD     = 384
BLOCK_SIZE = 256    # must be divisible by 16 for NVFP4 parity
BIAS       = False
DROPOUT    = 0.0

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE     = 64     # divisible by 16
MAX_ITERS      = 1000
EVAL_ITERS     = 50
EVAL_INTERVAL  = 100
LOG_INTERVAL   = 10

LR             = 1e-3
WEIGHT_DECAY   = 1e-1
BETA1, BETA2   = 0.9, 0.99
GRAD_CLIP      = 1.0
WARMUP_ITERS   = 100
LR_DECAY_ITERS = 1000
MIN_LR         = 1e-4

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
    print(f"Parameters: {model.get_num_params()/1e6:.2f}M")

    ctx       = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
    optimizer = model.configure_optimizers(WEIGHT_DECAY, LR, (BETA1, BETA2), 'cuda')

    torch.cuda.reset_peak_memory_stats()
    X, Y = get_batch('train')
    t_start = time.perf_counter()
    t0 = t_start
    iter_times = []
    final_val_loss = float('inf')

    for it in range(MAX_ITERS + 1):
        # LR update
        lr = cosine_lr(it)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # Eval
        if it % EVAL_INTERVAL == 0:
            losses = estimate_loss(model, ctx)
            print(f"step {it:4d}: train {losses['train']:.4f}  val {losses['val']:.4f}")
            final_val_loss = losses['val']

        if it == MAX_ITERS:
            break

        # Forward / backward
        with ctx:
            _, loss = model(X, Y)
        X, Y = get_batch('train')          # async prefetch
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        t1 = time.perf_counter()
        if it >= 10:                       # skip first 10 iters (JIT warmup)
            iter_times.append(t1 - t0)
        t0 = t1

        if it % LOG_INTERVAL == 0:
            print(f"  iter {it}: loss {loss.item():.4f}  lr {lr:.2e}")

    total_time    = time.perf_counter() - t_start
    iters_per_sec = 1.0 / (sum(iter_times) / len(iter_times)) if iter_times else 0.0
    peak_vram_mb  = torch.cuda.max_memory_allocated() / 1024**2

    print(f"\n{'='*60}")
    print(f"BF16 Training Complete")
    print(f"  Final val loss : {final_val_loss:.4f}  (PPL {math.exp(final_val_loss):.2f})")
    print(f"  Total time     : {total_time:.1f} s")
    print(f"  Iters/sec      : {iters_per_sec:.2f}")
    print(f"  Peak VRAM      : {peak_vram_mb:.1f} MB")
    print(f"{'='*60}\n")

    # ── Save checkpoint ────────────────────────────────────────────────────────
    ckpt_path = os.path.join(OUT_DIR, 'ckpt_bf16.pt')
    torch.save({
        'model':      model.state_dict(),
        'model_args': dict(n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD,
                           block_size=BLOCK_SIZE, bias=BIAS,
                           vocab_size=vocab_size, dropout=DROPOUT),
        'iter_num':   MAX_ITERS,
        'val_loss':   final_val_loss,
    }, ckpt_path)
    print(f"Checkpoint → {ckpt_path}")

    # ── Update results.json ────────────────────────────────────────────────────
    results_path = os.path.join(OUT_DIR, 'results.json')
    results = {}
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
    results['bf16'] = {
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
