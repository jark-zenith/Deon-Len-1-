# Experiment 002 — DEON-LEN 1 Beta Transformer Foundation

**Date:** 2026-09-28  
**Model:** DEON-LEN-1-beta-0.1  
**Status:** Architecture and training pipeline implemented; full training run requires the user's GPU environment.

## What changed from Alpha

Alpha was a hand-written NumPy character-level vanilla RNN with 9,836 parameters. Beta introduces a modern decoder-only Transformer foundation implemented directly in PyTorch.

### Beta components
- Byte-level BPE tokenizer using Hugging Face Tokenizers.
- Causal self-attention with PyTorch scaled dot-product attention when available.
- Pre-norm Transformer blocks.
- Learned token and positional embeddings.
- Weight-tied language-model head.
- AdamW optimizer and gradient clipping.
- CUDA mixed precision when a CUDA GPU is available.
- Train/validation split.
- Validation loss and perplexity logging.
- Portable PyTorch checkpoints.
- Standalone text generation script.

## Default Beta configuration
- 6 Transformer layers
- 256 hidden dimensions
- 8 attention heads
- 256-token context window
- BPE vocabulary target: 2,048
- Batch size: 16
- Learning rate: 3e-4
- Weight decay: 0.1
- Default training steps: 2,000

This is intentionally a small foundation model. The current Stage 1 corpus is far too small to produce a useful general-purpose assistant; it exists to validate the complete training architecture.

## Run

    pip install -r requirements.txt
    python -m beta.train --corpus stage1_corpus.txt --steps 2000
    python -m beta.generate --checkpoint beta/checkpoints/deon_beta.pt --prompt "DEON" --tokens 100

## Scientific interpretation

A successful run should show training loss decreasing. Validation loss should be monitored separately for overfitting. Perplexity is reported as exp(cross-entropy loss) and provides a standard next-token prediction metric.

## Important limitation

DEON-LEN Beta is an original implementation and training pipeline. It is not a copy of proprietary model weights, private training data, or internal systems. The eventual goal is comparable capability built from DEON's own architecture, data pipeline, instruction tuning, retrieval, memory, tools, and serving stack.

## Next milestone

**Beta.1 — Data and evaluation:** replace the proof-of-concept fable corpus with a substantially larger, legally usable corpus; add dataset quality checks, document deduplication, held-out evaluation data, checkpoint resume, and a benchmark suite before scaling model size.