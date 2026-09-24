# Experiment 001 — DEON-LEN 1 Alpha

**Date:** 2026-09-24
**Model version:** DEON-LEN-1-alpha-0.1
**Milestone target:** LEN 1 Alpha (pipeline works, model trains, checkpoints work, inference works, basic text generation works)

## Environment constraint

This environment has no internet access and no PyTorch installed (CPU-only,
1 core, NumPy available). The model was therefore implemented as a small
character-level vanilla RNN with hand-written forward pass and
backpropagation-through-time (BPTT) in raw NumPy. This is a real, trainable
neural network — not a wrapper around an external provider.

## Dataset (Stage 1)

- **Name:** DEON-LEN Stage 1 Proof-of-Concept Corpus v0.1
- **Source:** Original fable retellings authored directly for this project (no scraped or copyrighted text)
- **License:** Owned by Jark AI Tech Industries / Jark AI Tech Labs
- **Size:** 6,051 characters, ~1,100 words, 14 documents, 44 unique characters
- **Split:** 90% train (5,445 chars) / 10% validation (606 chars)
- Full provenance record: `data/raw/stage1_dataset_card.json`

## Architecture

- Character-level vanilla RNN
- Hidden size: 64
- Vocabulary size: 44 (character-level)
- Parameters: **9,836**
- Sequence length: 25 characters
- Optimizer: Adagrad, learning rate 0.1

## Training results

- Iterations: 4,000
- Training wall time: **6.18 seconds** (CPU, single core)
- Initial loss (near-random baseline, ln(44)): ~3.78
- **Final train loss: 2.0975**
- **Final val loss: 2.3705**

Loss curve (selected points):

| iter | train_loss | val_loss |
|------|-----------|----------|
| 0    | 3.7541    | 3.7431   |
| 800  | 2.3084    | 2.3769   |
| 2000 | 2.1271    | 2.3301   |
| 3800 | 2.1124    | 2.2976   |

Full curve: `training/train_log.json`

## Inference

Checkpoint saved to `checkpoints/len1_alpha.npz`, reloaded fresh in a
separate process (`inference/generate.py`), and used to generate text.
Inference latency: **~7ms** for 250 generated characters.

Sample output (prompt: "A fox"):

> A fox's des toraf, loy, wack andeed f clec the inge plom, in thosdelch
> ily oine tros atherghed athing laisk ot nockep tass ho soring beved
> viwhesther the wathemt of atoed in werinSed mit thicd cfeich tamperor
> hattiind the sass mos lous. The phet ing maton

**Honest assessment:** word-like shapes, spacing, and punctuation rhythm
are visibly learned; it is not fluent text. This is expected and correct
for a 9.8K-parameter model trained on 6KB of data for 6 seconds — it
proves the pipeline, not the product.

## Serving / cost accounting

`serving/serve.py` wraps inference in the full accounting schema required
by the monetization spec: request ID, model version, account/API key,
input/output/total tokens, request + compute duration, estimated compute
cost, customer charge, DEON revenue, gross margin, error status. Verified
working end-to-end — sample record in `monitoring/usage_log.jsonl`.

Example from a real run:
- 200 output tokens generated
- Compute duration: 5.5ms
- Estimated compute cost: $0.0000011
- Customer charge (at placeholder $0.02/1K output tokens): $0.004
- Gross margin: $0.0039989

These placeholder prices are configuration, not hard-coded — they live at
the top of `serving/serve.py` and should move into the admin-configurable
pricing table once DEON 1's billing system exists.

## Milestone checklist (LEN 1 Alpha, per training doc)

- [x] Pipeline works (data → tokenizer → model → train → checkpoint → eval → inference → serving)
- [x] Model trains (loss decreases from ~3.78 to ~2.10)
- [x] Checkpoints work (saved and reloaded in a separate process)
- [x] Inference works (~7ms per 250-char generation)
- [x] Basic text generation works (structurally plausible, not yet fluent)

**LEN 1 Alpha milestone: met**, on the smallest technically valid scale.

## What's needed for LEN 1 Beta

Per the roadmap, Beta requires: a larger/better dataset (Stage 2: DEON's
own docs/terminology), a real tokenizer (subword/BPE instead of raw
characters), improved training (more compute than a single CPU core
allows — GPU access via Colab/RunPod would meaningfully change what's
possible here), a proper evaluation suite (perplexity benchmarks, not
just loss), real API serving (an HTTP endpoint, not a Python function
call), and persistent usage accounting tied to real accounts.
