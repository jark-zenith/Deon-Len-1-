"""
DEON-LEN 1 Alpha - Model Architecture
--------------------------------------
A small vanilla character-level recurrent neural network, implemented
from scratch in NumPy (no external deep-learning framework was available
in this environment). This is deliberately the smallest architecture
that qualifies as a real, trainable neural language model:

    x_t (one-hot char) -> hidden state h_t -> output logits y_t -> softmax

Forward pass and backpropagation-through-time (BPTT) are both
implemented explicitly so every parameter update is inspectable.
This is a genuine LEN 1 "Alpha" milestone architecture, not a stand-in.
"""

import numpy as np


class DeonLenRNN:
    def __init__(self, vocab_size, hidden_size=64, seed=42):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        rng = np.random.default_rng(seed)

        # Xavier-ish small init
        self.Wxh = rng.standard_normal((hidden_size, vocab_size)) * 0.01
        self.Whh = rng.standard_normal((hidden_size, hidden_size)) * 0.01
        self.Why = rng.standard_normal((vocab_size, hidden_size)) * 0.01
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((vocab_size, 1))

    def param_count(self):
        return (
            self.Wxh.size + self.Whh.size + self.Why.size
            + self.bh.size + self.by.size
        )

    def forward_backward(self, inputs, targets, h_prev):
        """
        inputs, targets: lists of int token ids, same length (one training window)
        h_prev: (hidden_size, 1) initial hidden state
        Returns: loss, gradients dict, final hidden state
        """
        V, H = self.vocab_size, self.hidden_size
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(h_prev)
        loss = 0.0

        # forward pass
        for t in range(len(inputs)):
            xs[t] = np.zeros((V, 1))
            xs[t][inputs[t]] = 1
            hs[t] = np.tanh(self.Wxh @ xs[t] + self.Whh @ hs[t - 1] + self.bh)
            ys[t] = self.Why @ hs[t] + self.by
            # softmax
            exp_y = np.exp(ys[t] - np.max(ys[t]))
            ps[t] = exp_y / np.sum(exp_y)
            loss += -np.log(ps[t][targets[t], 0] + 1e-12)

        # backward pass (BPTT)
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)
        dh_next = np.zeros((H, 1))

        for t in reversed(range(len(inputs))):
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1
            dWhy += dy @ hs[t].T
            dby += dy
            dh = self.Why.T @ dy + dh_next
            dh_raw = (1 - hs[t] ** 2) * dh
            dbh += dh_raw
            dWxh += dh_raw @ xs[t].T
            dWhh += dh_raw @ hs[t - 1].T
            dh_next = self.Whh.T @ dh_raw

        for grad in (dWxh, dWhh, dWhy, dbh, dby):
            np.clip(grad, -5, 5, out=grad)

        grads = {"Wxh": dWxh, "Whh": dWhh, "Why": dWhy, "bh": dbh, "by": dby}
        return loss / len(inputs), grads, hs[len(inputs) - 1]

    def sample(self, h, seed_ix, n, rng=None):
        """Generate n token ids starting from hidden state h and seed token."""
        if rng is None:
            rng = np.random.default_rng()
        x = np.zeros((self.vocab_size, 1))
        x[seed_ix] = 1
        ids = []
        for _ in range(n):
            h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
            y = self.Why @ h + self.by
            exp_y = np.exp(y - np.max(y))
            p = (exp_y / np.sum(exp_y)).ravel()
            ix = rng.choice(range(self.vocab_size), p=p)
            x = np.zeros((self.vocab_size, 1))
            x[ix] = 1
            ids.append(ix)
        return ids

    def save(self, path):
        np.savez(
            path,
            Wxh=self.Wxh, Whh=self.Whh, Why=self.Why,
            bh=self.bh, by=self.by,
            vocab_size=self.vocab_size, hidden_size=self.hidden_size,
        )

    @classmethod
    def load(cls, path):
        data = np.load(path)
        model = cls(int(data["vocab_size"]), int(data["hidden_size"]))
        model.Wxh = data["Wxh"]
        model.Whh = data["Whh"]
        model.Why = data["Why"]
        model.bh = data["bh"]
        model.by = data["by"]
        return model
