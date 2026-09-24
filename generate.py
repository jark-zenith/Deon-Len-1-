"""
DEON-LEN 1 Alpha - Inference
------------------------------
Loads the saved checkpoint fresh (proving save/reload works) and
generates sample text. This is deliberately separate from training.py
per the architectural rule that the model be usable as an independent,
swappable component.
"""

import os
import sys
import time
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from tokenizer.tokenizer import CharTokenizer
from models.rnn_model import DeonLenRNN


def generate(prompt="A fox", max_new_chars=200, seed=None):
    tok = CharTokenizer.load(os.path.join(BASE, "tokenizer", "vocab.json"))
    model = DeonLenRNN.load(os.path.join(BASE, "checkpoints", "len1_alpha.npz"))

    rng = np.random.default_rng(seed)
    h = np.zeros((model.hidden_size, 1))

    # warm up hidden state on the prompt
    prompt_ids = tok.encode(prompt)
    for ch_id in prompt_ids[:-1]:
        x = np.zeros((model.vocab_size, 1))
        x[ch_id] = 1
        h = np.tanh(model.Wxh @ x + model.Whh @ h + model.bh)

    seed_ix = prompt_ids[-1]
    t0 = time.time()
    generated_ids = model.sample(h, seed_ix, max_new_chars, rng=rng)
    duration = time.time() - t0

    generated_text = prompt + tok.decode(generated_ids)
    return generated_text, duration, len(prompt_ids), len(generated_ids)


if __name__ == "__main__":
    text, duration, in_tokens, out_tokens = generate(prompt="A fox", max_new_chars=250, seed=7)
    print("--- DEON-LEN 1 Alpha sample generation ---")
    print(text)
    print("\n---")
    print(f"input tokens: {in_tokens} | output tokens: {out_tokens} | inference time: {duration*1000:.1f} ms")
