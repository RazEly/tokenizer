# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
bash init.sh          # installs uv, syncs deps, checks CUDA
```

Run all scripts from the `src/` root (not from inside `code/`).

## Key Commands

```bash
# Train all three tokenizers (reproducible, uses defaults)
uv run python generate_tokenizers.py

# Train a single tokenizer (for iteration/debugging)
uv run python code/train_tokenizer.py --domain_file data/domain_1_train.txt --output_dir tokenizers --vocab_size 5000

# Evaluate tokenizer: speed, efficiency, reconstruction
uv run python code/test_tokenizer.py --tokenizer_path trained_tokenizers/tokenizer_1.pkl --train_file data/domain_1_train.txt --test_file data/domain_1_dev.txt

# Train NER model on domain 1
uv run python code/train_ner_model.py --tokenizer_path trained_tokenizers/tokenizer_1.pkl --train_file data/ner_data/train_1_binary.tagged --dev_file data/ner_data/dev_1_binary.tagged

# Train NER model on domain 2
uv run python code/train_ner_model.py --tokenizer_path trained_tokenizers/tokenizer_2.pkl --train_file data/ner_data/train_2_binary.tagged --dev_file data/ner_data/dev_2_binary.tagged

# Validate submission zip
uv run python check_submission.py
```

## Architecture

### The only file to edit: `code/bpe_tokenizer.py`

`BPETokenizer` extends `BaseTokenizer` (`code/base_tokenizer.py`). The grader unpickles trained tokenizers using only `BaseTokenizer` — the class does **not** need to be importable, just picklable. `BaseTokenizer.save()` / `.load()` use `pickle.dump` / `pickle.load`.

### Do NOT edit or submit

`generate_tokenizers.py`, `check_submission.py`, `base_tokenizer.py`, `train_ner_model.py`. The grader runs the originals.

### Two-stage BPE (`BPETokenizer.train`)

**Stage A** (`_train_within_word`): Classic within-word BPE. Uses GPT-2 pre-tokenization regex to split text into chunks, then byte-encodes each chunk. Runs an incremental max-heap over pair frequencies for O(log n) merge selection. Returns `line_keys` (pre-computed chunk keys per line) so Stage B avoids a second regex pass.

**Stage B** (`_train_bigrams`): Merges adjacent *atomic whole-word* pairs (words that Stage A compressed to a single token, containing at least one letter) into cross-word bigram tokens. Hard guarantee: synthesizes at least one bigram if none emerge naturally (assignment requirement).

### Encoding pipeline (`encode`)

1. Pre-tokenize text with `pat.findall` → chunks
2. `_encode_chunk` applies Stage A BPE merges (cached per chunk string)
3. Linear scan over chunks applies Stage B bigram merges (each atomic adjacent pair)
4. Map symbols to IDs; unknown symbols → `[UNK]` (id 1)

### NER alignment (`NERDataset` in `train_ner_model.py`)

Token-to-word alignment uses **first-subtoken labeling**: only the first token of each word gets the word's NER label; others get `-100` (ignored in loss). The fast path calls `tokenizer.encode_with_offsets(text)` if the method exists — implement this for a significant speed-up over the fallback O(n²) decode loop.

### Constraints

- `space_token` attribute required on the class (grader and NERDataset check for it). Current impl: `self.space_token = self.byte_encoder[ord(" ")]` → `'Ġ'`.
- Must produce ≥ 1 bigram token or the submission is disqualified.
- NER model hyperparameters are **fixed** (SEED=42, BATCH=32, LR=0.01, EPOCHS=20) — do not pass different values.
- Python `>=3.11,<3.13`. Torch pulled from PyTorch's cu126 index for Tesla M60 GPU compatibility.

### Output artifacts

Tokenizers saved to `trained_tokenizers/tokenizer_{1,2,3}.pkl`. Submission zip must match exact structure (see `check_submission.py`).
