#!/usr/bin/env python3
"""
Optuna-based hyperparameter tuning for BPE tokenizer — domains 1 and 2.

Uses TPE (Tree-structured Parzen Estimator) to search a continuous space more
efficiently than grid search. Optimizes F1 as the Optuna objective; composite
score (F1 + speed + time) computed post-hoc across all completed trials.

Domain 3 is the *hidden* eval domain: it can't be tuned directly (no text, no
labels), so it trains on the combined domain_1+domain_2 corpus and is scored on
the MEAN F1 across both dev sets — cross-domain transfer as a proxy for the
unknown domain. The `strip_handles` hyperparameter (drop @mentions before
training) is searched so the composite reveals whether stripping actually helps.

Usage:
    uv run python tune_tokenizers.py                         # all 3 domains, 30 trials each (90 total)
    uv run python tune_tokenizers.py --domain both           # domains 1 & 2 only
    uv run python tune_tokenizers.py --domain 3 --n_trials 15 --tune_epochs 5
    uv run python tune_tokenizers.py --f1_weight 0.8 --speed_weight 0.1 --time_weight 0.1
    uv run python tune_tokenizers.py --sampler random        # random search baseline

Estimated runtime (GPU): ~5 min/trial × 30 trials × 3 domains ≈ 7.5 hours.
Use --n_trials 15 --tune_epochs 5 for a ~1-hour exploratory pass.

Results saved to:
    <output_dir>/domain_<N>_all.json   — all trial results + composite scores
    <output_dir>/domain_<N>_best.json  — best params ready to apply
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent / "code"))

from bpe_tokenizer import BPETokenizer, HANDLE_RE
from train_ner_model import (
    NERDataset,
    NERModel,
    collate_fn,
    evaluate_model,
    read_ner_data,
    set_seed,
    SEED,
    BATCH_SIZE,
    LEARNING_RATE,
)

DEFAULT_N_TRIALS    = 30
DEFAULT_TUNE_EPOCHS = 8    # fewer than prod (20); captures peak F1 region
SPEED_SAMPLE_CHARS  = 50_000

# Unified schema: train_texts = corpus file(s) concatenated for tokenizer training;
# ner_pairs = (train, dev) tagged-file pairs the NER model is scored on. Domain 3
# trains on the combined corpus and is scored on BOTH dev sets (cross-domain proxy
# for the hidden eval domain — we can't see it, so transfer across the two known
# domains is the best available signal).
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


# ── Tokenizer factory ─────────────────────────────────────────────────────── #

def make_tokenizer(params, texts):
    """Build BPETokenizer with explicit params, bypassing _auto_configure."""
    tok = BPETokenizer(vocab_size=params["vocab_size"])
    tok._bigram_reserve_frac = params["bigram_reserve_frac"]
    tok._lbpe_exp             = params["lbpe_exp"]
    tok._min_pair_freq        = params["min_pair_freq"]
    tok._min_bigram_freq      = params["min_bigram_freq"]

    # Instance-dict lookup beats class method; plain functions are not descriptors
    # so 'self' is not injected — _auto_configure(texts) calls lambda(texts).
    tok._auto_configure = lambda _texts: None
    return tok


# ── Benchmarks ────────────────────────────────────────────────────────────── #

def measure_speed(tok, texts):
    blob = " ".join(texts)[:SPEED_SAMPLE_CHARS]
    tok.encode(blob[:500])       # warm
    tok._encode_cache = {}
    tok.cache         = {}
    t0   = time.perf_counter()
    toks = tok.encode(blob)
    dt   = time.perf_counter() - t0
    return len(toks) / dt if dt > 0 else float("inf")


def train_and_evaluate_ner(tok, ner_train, ner_dev, num_epochs, device):
    """Returns (best_dev_f1, wall_time_sec). Uses same arch/hyperparams as train_ner_model.py."""
    set_seed(SEED)

    train_texts, train_labels = read_ner_data(ner_train)
    dev_texts,   dev_labels   = read_ner_data(ner_dev)

    train_ds = NERDataset(train_texts, train_labels, tok)
    dev_ds   = NERDataset(dev_texts,   dev_labels,   tok)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn)
    dev_dl   = DataLoader(dev_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model     = NERModel(tok.get_vocab_size(), num_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn   = nn.CrossEntropyLoss(ignore_index=-100)

    best_f1 = 0.0
    t0      = time.perf_counter()

    for epoch in range(num_epochs):
        model.train()
        for batch in train_dl:
            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)
            optimizer.zero_grad()
            logits    = model(input_ids)
            B, T, C   = logits.shape
            loss_fn(logits.view(-1, C), labels.view(-1)).backward()
            optimizer.step()

        metrics = evaluate_model(model, dev_dl, device)
        best_f1 = max(best_f1, metrics["f1"])
        print(f"    epoch {epoch+1}/{num_epochs}  F1={metrics['f1']:.4f}  (best={best_f1:.4f})")

    return best_f1, time.perf_counter() - t0


# ── Optuna objective ──────────────────────────────────────────────────────── #

def make_objective(train_texts, cfg, tune_epochs, device):
    """Returns an Optuna objective closure over the domain data."""

    def objective(trial):
        params = {
            # Larger vocab → more whole-word tokens → cleaner NER alignment.
            # 20k is the practical ceiling before training time dominates.
            "vocab_size": trial.suggest_categorical(
                "vocab_size", [5000, 8000, 10000, 12000, 15000, 20000]
            ),
            # Log scale: the difference 0.001→0.01 matters more than 0.10→0.15.
            # Lower floor lets Optuna explore near-zero bigram budgets (best for NER).
            "bigram_reserve_frac": trial.suggest_float(
                "bigram_reserve_frac", 0.001, 0.20, log=True
            ),
            # 0 = pure frequency, 1 = full LBPE, >1 = super length-biased.
            # Allows Optuna to explore beyond the "standard" LBPE upper bound.
            "lbpe_exp": trial.suggest_float("lbpe_exp", 0.0, 1.5),
            # Higher ceiling for large corpora where freq-2 pairs are still noise.
            "min_pair_freq": trial.suggest_int("min_pair_freq", 1, 10),
            # Higher ceiling: aggressively filter rare cross-word tokens for NER.
            "min_bigram_freq": trial.suggest_int("min_bigram_freq", 1, 20),
            # Strip @handles from the training corpus? Searched so the composite
            # tells us whether handle-stripping (the domain-3 flow) actually helps.
            "strip_handles": trial.suggest_categorical("strip_handles", [True, False]),
        }

        print(f"\n  Trial {trial.number}: {params}")

        # @handle stripping is a training-corpus transform (encode() is unchanged).
        texts = train_texts
        if params["strip_handles"]:
            texts = [HANDLE_RE.sub("", t) for t in train_texts]

        tok = make_tokenizer(params, texts)
        t0  = time.perf_counter()
        tok.train(texts)
        tok_time = time.perf_counter() - t0
        print(f"  tokenizer: {tok_time:.1f}s  vocab={tok.get_vocab_size():,}  bigrams={len(tok.bigram_merges)}")

        speed = measure_speed(tok, texts)
        print(f"  speed: {speed:,.0f} tok/s")

        # Score NER on every (train, dev) pair; combined F1 = mean across domains.
        f1s, ner_time = [], 0.0
        for ner_train, ner_dev in cfg["ner_pairs"]:
            f1_i, t_i = train_and_evaluate_ner(tok, ner_train, ner_dev, tune_epochs, device)
            f1s.append(f1_i)
            ner_time += t_i
        f1 = sum(f1s) / len(f1s)
        total_time = tok_time + ner_time
        print(f"  F1 per domain: {[f'{x:.4f}' for x in f1s]}  mean={f1:.4f}  total: {total_time:.1f}s")

        trial.set_user_attr("speed",        speed)
        trial.set_user_attr("total_time",   total_time)
        trial.set_user_attr("n_bigrams",    len(tok.bigram_merges))
        trial.set_user_attr("f1_per_domain", f1s)

        return f1  # Optuna maximizes mean F1; composite computed post-hoc

    return objective


# ── Post-hoc composite scoring ─────────────────────────────────────────────── #

def normalize(vals, higher_better=True):
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return [1.0] * len(vals)
    normed = [(v - lo) / (hi - lo) for v in vals]
    return normed if higher_better else [1.0 - n for n in normed]


def compute_composite(study, weights):
    """
    Normalize F1, speed, and time across all completed trials, then compute
    weighted composite. Returns list of (trial, composite_score) sorted descending.
    """
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        return []

    f1s    = [t.value                    for t in completed]
    speeds = [t.user_attrs["speed"]      for t in completed]
    times  = [t.user_attrs["total_time"] for t in completed]

    f1_n    = normalize(f1s,    higher_better=True)
    speed_n = normalize(speeds, higher_better=True)
    time_n  = normalize(times,  higher_better=False)

    scored = []
    for trial, fn, sn, tn in zip(completed, f1_n, speed_n, time_n):
        score = weights["f1"] * fn + weights["speed"] * sn + weights["time"] * tn
        scored.append((trial, score))

    scored.sort(key=lambda x: -x[1])
    return scored


# ── Per-domain tuning ─────────────────────────────────────────────────────── #

def tune_domain(domain_id, cfg, weights, n_trials, tune_epochs, sampler_name, output_dir, device):
    print(f"\n{'='*62}")
    print(f"  Domain {domain_id}  |  {n_trials} trials  |  {tune_epochs} NER epochs/trial")
    print(f"{'='*62}")

    train_texts = []
    for p in cfg["train_texts"]:
        train_texts.extend(Path(p).read_text(encoding="utf-8").splitlines())
    print(f"  Corpus: {len(train_texts):,} lines from {len(cfg['train_texts'])} file(s)")

    if sampler_name == "random":
        sampler = optuna.samplers.RandomSampler()
    else:
        sampler = optuna.samplers.TPESampler(seed=SEED)

    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        make_objective(train_texts, cfg, tune_epochs, device),
        n_trials=n_trials,
    )

    scored = compute_composite(study, weights)
    if not scored:
        print("No completed trials.")
        return None

    best_trial, best_composite = scored[0]

    print(f"\n{'─'*62}")
    print(f"Best for domain {domain_id} (by composite score):")
    for k, v in best_trial.params.items():
        print(f"  {k:<22} = {v}")
    print(f"  F1={best_trial.value:.4f}  "
          f"speed={best_trial.user_attrs['speed']:,.0f} tok/s  "
          f"composite={best_composite:.4f}")

    # Serialise all results
    all_results = []
    for trial, cscore in scored:
        all_results.append({
            "trial_number":         trial.number,
            "params":               trial.params,
            "f1":                   trial.value,
            "f1_per_domain":        trial.user_attrs["f1_per_domain"],
            "speed_toks_per_sec":   trial.user_attrs["speed"],
            "total_train_time_sec": trial.user_attrs["total_time"],
            "n_bigrams":            trial.user_attrs["n_bigrams"],
            "composite_score":      cscore,
        })

    os.makedirs(output_dir, exist_ok=True)
    all_path  = Path(output_dir) / f"domain_{domain_id}_all.json"
    best_path = Path(output_dir) / f"domain_{domain_id}_best.json"

    with open(all_path, "w") as f:
        json.dump({
            "domain":      domain_id,
            "n_trials":    n_trials,
            "tune_epochs": tune_epochs,
            "sampler":     sampler_name,
            "weights":     weights,
            "results":     all_results,
        }, f, indent=2)

    with open(best_path, "w") as f:
        json.dump({
            "domain":  domain_id,
            **best_trial.params,
            "scores": {
                "f1":                   best_trial.value,
                "f1_per_domain":        best_trial.user_attrs["f1_per_domain"],
                "speed_toks_per_sec":   best_trial.user_attrs["speed"],
                "total_train_time_sec": best_trial.user_attrs["total_time"],
            },
            "composite_score": best_composite,
            "weights":         weights,
        }, f, indent=2)

    print(f"  → {best_path}")
    print(f"  → {all_path}")
    return best_trial


# ── Entry point ───────────────────────────────────────────────────────────── #

def main():
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter search for BPE tokenizer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--domain",       choices=["1", "2", "3", "both", "all"], default="all")
    parser.add_argument("--output_dir",   default="tuning_results")
    parser.add_argument("--n_trials",     type=int, default=DEFAULT_N_TRIALS,
                        help="Optuna trials per domain")
    parser.add_argument("--tune_epochs",  type=int, default=DEFAULT_TUNE_EPOCHS,
                        help="NER epochs per trial (5 = fast pass, 8 = default, 12 = thorough)")
    parser.add_argument("--sampler",      choices=["tpe", "random"], default="tpe",
                        help="tpe = Bayesian (recommended), random = baseline")
    parser.add_argument("--f1_weight",    type=float, default=0.70)
    parser.add_argument("--speed_weight", type=float, default=0.20)
    parser.add_argument("--time_weight",  type=float, default=0.10)
    args = parser.parse_args()

    weights = {"f1": args.f1_weight, "speed": args.speed_weight, "time": args.time_weight}
    if abs(sum(weights.values()) - 1.0) > 1e-6:
        parser.error("Weights must sum to 1.0")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:   {device}")
    print(f"Sampler:  {args.sampler.upper()}  |  trials: {args.n_trials}/domain  |  NER epochs: {args.tune_epochs}")
    print(f"Weights:  F1={weights['f1']}  speed={weights['speed']}  time={weights['time']}")

    if args.domain == "both":
        domains = [1, 2]
    elif args.domain == "all":
        domains = [1, 2, 3]
    else:
        domains = [int(args.domain)]
    for d in domains:
        tune_domain(
            d, DOMAINS[d], weights,
            args.n_trials, args.tune_epochs, args.sampler,
            args.output_dir, device,
        )

    print("\nDone. Load tuning_results/domain_<N>_best.json and set params in BPETokenizer.__init__.")


if __name__ == "__main__":
    main()
