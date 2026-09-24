"""
DEON-LEN 1 Alpha - Training Script
------------------------------------
Trains the small character-level RNN on the Stage 1 corpus, using
Adagrad-style adaptive updates (standard for small char-RNNs). Logs
training and validation loss at regular intervals and writes:
  - a checkpoint (models/checkpoints/len1_alpha.npz)
  - a structured training log (training/train_log.json)
"""

import os
import sys
import json
import time
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from tokenizer.tokenizer import CharTokenizer
from models.rnn_model import DeonLenRNN


def main():
    t_start = time.time()

    corpus_path = os.path.join(BASE, "data", "raw", "stage1_corpus.txt")
    text = open(corpus_path).read()

    tok = CharTokenizer.load(os.path.join(BASE, "tokenizer", "vocab.json"))
    data_ids = tok.encode(text)

    # 90/10 train/val split
    split = int(0.9 * len(data_ids))
    train_ids = data_ids[:split]
    val_ids = data_ids[split:]

    seq_len = 25
    hidden_size = 64
    learning_rate = 0.1
    max_iters = 4000
    eval_every = 200

    model = DeonLenRNN(vocab_size=tok.vocab_size, hidden_size=hidden_size, seed=42)
    param_count = model.param_count()

    # Adagrad accumulators
    mem = {k: np.zeros_like(v) for k, v in {
        "Wxh": model.Wxh, "Whh": model.Whh, "Why": model.Why,
        "bh": model.bh, "by": model.by,
    }.items()}

    def eval_loss(ids, n_windows=20):
        """Average loss over a handful of windows, without updating weights."""
        if len(ids) <= seq_len + 1:
            n_windows = 1
        rng = np.random.default_rng(0)
        losses = []
        h = np.zeros((hidden_size, 1))
        max_start = max(1, len(ids) - seq_len - 1)
        for _ in range(n_windows):
            p = int(rng.integers(0, max_start))
            inputs = ids[p:p + seq_len]
            targets = ids[p + 1:p + seq_len + 1]
            if len(inputs) < 2:
                continue
            loss, _, _ = model.forward_backward(inputs, targets, h)
            losses.append(loss)
        return float(np.mean(losses)) if losses else float("nan")

    log = {
        "model_version": "DEON-LEN-1-alpha-0.1",
        "architecture": "char-level vanilla RNN (NumPy, hand-written BPTT)",
        "hidden_size": hidden_size,
        "vocab_size": tok.vocab_size,
        "param_count": int(param_count),
        "seq_len": seq_len,
        "learning_rate": learning_rate,
        "max_iters": max_iters,
        "train_chars": len(train_ids),
        "val_chars": len(val_ids),
        "loss_curve": [],
    }

    n, p = 0, 0
    h_prev = np.zeros((hidden_size, 1))
    smooth_loss = -np.log(1.0 / tok.vocab_size) * seq_len  # initial expected loss

    while n < max_iters:
        if p + seq_len + 1 >= len(train_ids) or n == 0:
            h_prev = np.zeros((hidden_size, 1))
            p = 0

        inputs = train_ids[p:p + seq_len]
        targets = train_ids[p + 1:p + seq_len + 1]

        loss, grads, h_prev = model.forward_backward(inputs, targets, h_prev)
        smooth_loss = smooth_loss * 0.999 + loss * 0.001

        # Adagrad update
        params = {"Wxh": model.Wxh, "Whh": model.Whh, "Why": model.Why,
                  "bh": model.bh, "by": model.by}
        for k in params:
            mem[k] += grads[k] ** 2
            params[k] -= learning_rate * grads[k] / (np.sqrt(mem[k]) + 1e-8)

        if n % eval_every == 0:
            val_l = eval_loss(val_ids)
            train_l = eval_loss(train_ids)
            log["loss_curve"].append({
                "iter": n, "train_loss": round(train_l, 4),
                "val_loss": round(val_l, 4), "smooth_train_loss": round(float(smooth_loss), 4),
            })
            print(f"iter {n:5d} | train_loss {train_l:.4f} | val_loss {val_l:.4f}")

        p += seq_len
        n += 1

    final_val = eval_loss(val_ids, n_windows=40)
    final_train = eval_loss(train_ids, n_windows=40)
    log["final_train_loss"] = round(final_train, 4)
    log["final_val_loss"] = round(final_val, 4)
    log["training_wall_time_seconds"] = round(time.time() - t_start, 2)

    ckpt_dir = os.path.join(BASE, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "len1_alpha.npz")
    model.save(ckpt_path)
    log["checkpoint_path"] = ckpt_path

    with open(os.path.join(BASE, "training", "train_log.json"), "w") as f:
        json.dump(log, f, indent=2)

    print("\n--- DEON-LEN 1 Alpha training complete ---")
    print(f"Parameters: {param_count}")
    print(f"Final train loss: {final_train:.4f} | Final val loss: {final_val:.4f}")
    print(f"Wall time: {log['training_wall_time_seconds']}s")
    print(f"Checkpoint saved to: {ckpt_path}")


if __name__ == "__main__":
    main()
