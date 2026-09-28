# DEON-LEN 1 Beta — Transformer Foundation

Beta replaces the Alpha NumPy character RNN with a GPT-style decoder-only Transformer in PyTorch.

## Architecture
- Byte-level BPE tokenizer
- Causal self-attention
- Pre-norm Transformer blocks
- Learned token + positional embeddings
- Weight-tied language-model head
- AdamW optimization
- GPU mixed precision when CUDA is available
- Validation loss + perplexity
- Checkpoint save/load

## First run
```bash
pip install -r requirements.txt
python -m beta.train --corpus stage1_corpus.txt --steps 2000
python -m beta.generate --checkpoint beta/checkpoints/deon_beta.pt --prompt "DEON"
```

The Stage 1 corpus is intentionally tiny and is only suitable for proving the training pipeline. A useful language model requires a much larger, legally usable corpus and substantially more compute.

## Important
Alpha remains the historical proof-of-concept. Beta is the new model foundation; it is not a literal copy of proprietary ChatGPT weights or training data.
