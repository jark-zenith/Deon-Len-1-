"""
DEON-LEN 1 Tokenizer (Alpha)
----------------------------
A minimal character-level tokenizer. This is intentionally the simplest
possible tokenizer so that the full pipeline (data -> vocab -> train ->
eval -> inference -> serving) can be proven end-to-end before investing
in a more sophisticated subword tokenizer (e.g. BPE) for LEN 1 Beta.
"""

import json
import os


class CharTokenizer:
    def __init__(self, chars):
        self.chars = sorted(list(chars))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    @classmethod
    def build_from_text(cls, text):
        unique_chars = set(text)
        return cls(unique_chars)

    def encode(self, text):
        return [self.stoi[ch] for ch in text]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"chars": self.chars}, f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = json.load(f)
        return cls(data["chars"])


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    corpus_path = os.path.join(base, "data", "raw", "stage1_corpus.txt")
    text = open(corpus_path).read()

    tok = CharTokenizer.build_from_text(text)
    out_path = os.path.join(base, "tokenizer", "vocab.json")
    tok.save(out_path)

    print(f"Vocab size: {tok.vocab_size}")
    print(f"Saved vocabulary to: {out_path}")
    sample = text[:40]
    encoded = tok.encode(sample)
    decoded = tok.decode(encoded)
    print(f"Round-trip check -> matches original: {decoded == sample}")
