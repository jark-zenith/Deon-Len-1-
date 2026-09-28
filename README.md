# DEON-LEN 1

DEON-LEN 1 is Jark's experimental language-model project, progressing from a transparent NumPy RNN proof-of-concept toward a modern Transformer-based foundation.

## Current status

### Alpha — complete proof of concept
- Character-level vanilla RNN implemented from scratch in NumPy.
- Training, validation, checkpointing, generation, HTTP serving, and usage accounting implemented.
- Historical Alpha files remain at repository root.

### Beta — Transformer foundation implemented
- PyTorch decoder-only Transformer.
- Byte-level BPE tokenizer.
- GPU/mixed-precision training support.
- Validation loss and perplexity.
- Checkpointing and generation.

See `beta/README.md` and `experiment_002_len1_beta.md`.

## Quick start

    pip install -r requirements.txt
    python -m beta.train --corpus stage1_corpus.txt --steps 2000
    python -m beta.generate --checkpoint beta/checkpoints/deon_beta.pt --prompt "DEON"

The current corpus is a tiny engineering test set, not a production pretraining corpus. Scaling capability requires better data, much more compute, evaluation, and later instruction tuning.

## Project direction

DEON is being developed as both a language model and, later, an assistant platform: model weights + tokenizer + training + instruction tuning + inference + memory/retrieval + tools + API infrastructure.