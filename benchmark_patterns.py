#!/usr/bin/env python3
"""
A/B benchmark: pre-tokenization regex variants on a FIXED hyperparam config.

Holds every BPE hyperparam constant (the production per-domain defaults set by
`BPETokenizer._auto_configure`) and varies ONLY the pre-tokenization regex, so the
reported deltas isolate the pattern change. Per pattern it reports:

    train_s      tokenizer fit time (s)
    chars/tok    compression on dev text (higher = better)
    tok/s        encode throughput
    f1           NER dev F1 (skip with --no-ner for a fast CPU pass)

Candidate patterns live HERE, not in bpe_tokenizer.py: they are experimental until
this benchmark justifies promoting one to the graded tokenizer's default.

Usage:
    uv run python benchmark_patterns.py --domain 1
    uv run python benchmark_patterns.py --domain 1 --patterns current cl100k o200k
    uv run python benchmark_patterns.py --domain 1 --no-ner          # fast, CPU only
    uv run python benchmark_patterns.py --domain all --epochs 8
"""

import argparse
import sys
import time
from pathlib import Path

import regex as re

sys.path.insert(0, str(Path(__file__).parent / "code"))

from bpe_tokenizer import BPETokenizer, HANDLE_RE, PRETOKENIZE_PATTERN

# ── Candidate patterns ────────────────────────────────────────────────────── #
# current  = raw GPT-2 (production default, imported from bpe_tokenizer)
# cl100k   = GPT-3.5/4 — adds possessive quantifiers (linear-time, no catastrophic
#            backtracking) + tighter whitespace handling. English boundaries ≈ GPT-2.
# o200k    = GPT-4o — case-aware word splitting (camelCase/hashtags split at case
#            boundary) + multilingual letter classes (\p{Lo}/\p{Lm}/\p{M}).
PATTERNS = {
    "current": PRETOKENIZE_PATTERN,
    "cl100k": r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s""",
    "o200k": r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+""",
}

# Corpus + NER eval pairs per domain (mirrors tune_tokenizers.DOMAINS). Domain 3
# trains on the combined corpus and is scored on the mean of both dev sets.
DOMAINS = {
    1: {
        "train_texts": ["data/domain_1_train.txt"],
        "ner_pairs":   [("data/ner_data/train_1_binary.tagged",
                         "data/ner_data/dev_1_binary.tagged")],
    },
    2: {
        "train_texts": ["data/domain_2_train.txt"],
        "ner_pairs":   [("data/ner_data/train_2_binary.tagged",
                         "data/ner_data/dev_2_binary.tagged")],
    },
    3: {
        "train_texts": ["data/domain_1_train.txt", "data/domain_2_train.txt"],
        "ner_pairs":   [("data/ner_data/train_1_binary.tagged",
                         "data/ner_data/dev_1_binary.tagged"),
                        ("data/ner_data/train_2_binary.tagged",
                         "data/ner_data/dev_2_binary.tagged")],
    },
}

SPEED_SAMPLE_CHARS = 50_000


def build(pattern_name, texts):
    """Train a tokenizer with the named pattern; production hyperparams untouched.

    _auto_configure (called inside train) sets the per-domain hyperparams and no
    longer touches .pat, so the pattern we assign here survives training.
    """
    tok = BPETokenizer(vocab_size=5000)
    tok.pat = re.compile(PATTERNS[pattern_name], re.UNICODE)
    t0 = time.perf_counter()
    tok.train(texts)
    return tok, time.perf_counter() - t0


def measure_speed(tok, texts):
    blob = " ".join(texts)[:SPEED_SAMPLE_CHARS]
    tok.encode(blob[:500])              # warm caches
    tok._encode_cache = {}
    tok.cache = {}
    t0 = time.perf_counter()
    toks = tok.encode(blob)
    dt = time.perf_counter() - t0
    return len(toks) / dt if dt > 0 else float("inf")


def measure_compression(tok, texts):
    """chars / token over a dev sample (higher = fewer tokens per char = better)."""
    sample = [t for t in texts if t.strip()][:5000]
    n_chars = sum(len(t) for t in sample)
    n_toks = sum(len(tok.encode(t)) for t in sample)
    return n_chars / n_toks if n_toks else 0.0


def run_domain(domain_id, pattern_names, epochs, do_ner):
    cfg = DOMAINS[domain_id]
    train_texts = []
    for p in cfg["train_texts"]:
        train_texts.extend(Path(p).read_text(encoding="utf-8").splitlines())

    # Dev text for compression = first NER dev file's raw sentences.
    dev_path = cfg["ner_pairs"][0][1]

    device = None
    train_eval = None
    if do_ner:
        import torch
        from train_ner_model import set_seed  # noqa: F401  (seeded inside helper)
        from tune_tokenizers import train_and_evaluate_ner as train_eval
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Compression dev text: strip NER tag columns -> plain sentences.
    dev_sents = []
    for line in Path(dev_path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            dev_sents.append(line.split("\t")[0])

    print(f"\n{'='*72}")
    print(f"  Domain {domain_id}  |  {len(train_texts):,} train lines  |  NER={'on' if do_ner else 'off'}")
    print(f"{'='*72}")
    header = f"  {'pattern':<9} {'train_s':>8} {'chars/tok':>10} {'tok/s':>12}"
    if do_ner:
        header += f" {'f1':>8}"
    print(header)

    rows = []
    for name in pattern_names:
        # Domain-3 strips @handles before training (production "mixed" flow).
        texts = train_texts
        if domain_id == 3:
            texts = [HANDLE_RE.sub("", t) for t in train_texts]

        tok, train_s = build(name, texts)
        cpt = measure_compression(tok, dev_sents)
        spd = measure_speed(tok, texts)

        f1 = None
        if do_ner:
            f1s = []
            for ner_train, ner_dev in cfg["ner_pairs"]:
                f1_i, _ = train_eval(tok, ner_train, ner_dev, epochs, device)
                f1s.append(f1_i)
            f1 = sum(f1s) / len(f1s)

        line = f"  {name:<9} {train_s:>8.1f} {cpt:>10.3f} {spd:>12,.0f}"
        if do_ner:
            line += f" {f1:>8.4f}"
        print(line)
        rows.append((name, train_s, cpt, spd, f1))

    # Deltas vs current.
    base = next((r for r in rows if r[0] == "current"), None)
    if base:
        print(f"\n  Δ vs current  (chars/tok, tok/s{', f1' if do_ner else ''}):")
        for name, _, cpt, spd, f1 in rows:
            if name == "current":
                continue
            d = f"    {name:<9} {cpt-base[2]:+.3f}  {spd-base[3]:+,.0f}"
            if do_ner and f1 is not None and base[4] is not None:
                d += f"  {f1-base[4]:+.4f}"
            print(d)
    return rows


def main():
    ap = argparse.ArgumentParser(
        description="A/B pre-tokenization regex variants (hyperparams fixed)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--domain", choices=["1", "2", "3", "all"], default="1")
    ap.add_argument("--patterns", nargs="+", default=["current", "cl100k", "o200k"],
                    choices=list(PATTERNS))
    ap.add_argument("--epochs", type=int, default=8, help="NER epochs per pattern")
    ap.add_argument("--no-ner", action="store_true",
                    help="skip NER F1 (compression + speed only; CPU, fast)")
    args = ap.parse_args()

    domains = [1, 2, 3] if args.domain == "all" else [int(args.domain)]
    for d in domains:
        run_domain(d, args.patterns, args.epochs, not args.no_ner)


if __name__ == "__main__":
    main()
