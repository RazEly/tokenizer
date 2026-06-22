"""
Reproducibly train and save all three tokenizers for the assignment.

This is a course-provided script: you do NOT edit or submit it. Implement your
tokenizer in code/bpe_tokenizer.py (class `BPETokenizer`, subclassing
BaseTokenizer); this script imports it and trains the three tokenizers.

Run it from the `For Students/` directory:

    uv run python generate_tokenizers.py

All behaviour is configurable via command-line arguments (see --help); the
defaults reproduce the standard setup. Tokenizers 1 and 2 are evaluated on NER
domains 1 and 2; tokenizer 3 is evaluated on a *hidden* domain.
"""

import argparse
import os
import random
import sys

import numpy as np

# This script lives at the root; the tokenizer modules live in code/.
# Put code/ on the path before importing the student's tokenizer.
CODE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "code")
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from bpe_tokenizer import BPETokenizer as Tokenizer

# Defaults (override via CLI).
DEFAULT_DATA_DIR = "data"
DEFAULT_OUTPUT_DIR = "trained_tokenizers"
DEFAULT_VOCAB_SIZE = 5000
DEFAULT_SEED = 42

# Default training files per tokenizer (filenames, joined to --data_dir).
DEFAULT_TRAIN_FILES = {
    1: ["domain_1_train.txt"],
    2: ["domain_2_train.txt"],
    # Naive baseline: both given domains combined. The hidden test domain
    # differs from both -- a smarter data mix / strategy may do better.
    3: ["domain_1_train.txt", "domain_2_train.txt"],
}


def set_seed(seed: int) -> None:
    """Seed RNGs so tokenizer training is reproducible."""
    random.seed(seed)
    np.random.seed(seed)


def read_text_file(path: str):
    """Read all lines from a text file (one sentence per line)."""
    with open(path, "r", encoding="utf-8") as f:
        return f.readlines()


def read_text_files(paths):
    """Read and concatenate lines from several text files."""
    texts = []
    for path in paths:
        texts.extend(read_text_file(path))
    return texts


def train_and_save(tokenizer, texts, out_path: str) -> None:
    """Train a tokenizer on `texts` and save it to `out_path`."""
    print(f"\nTraining {type(tokenizer).__name__} on {len(texts)} lines -> {out_path}")
    tokenizer.train(texts)
    tokenizer.save(out_path)  # BaseTokenizer.save() creates the output directory
    print(f"  Saved {out_path} with vocab size {tokenizer.get_vocab_size()}")


def main(args) -> None:
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    train_files = {1: args.train_files_1, 2: args.train_files_2, 3: args.train_files_3}
    for index in (1, 2, 3):
        paths = [os.path.join(args.data_dir, name) for name in train_files[index]]
        texts = read_text_files(paths)
        out_path = os.path.join(args.output_dir, f"tokenizer_{index}.pkl")
        train_and_save(Tokenizer(vocab_size=args.vocab_size), texts, out_path)

    print("\nDone. All three tokenizers written to", args.output_dir)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Reproducibly train and save tokenizer_{1,2,3}.pkl."
    )
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR,
                        help="Directory containing the domain_*_train.txt files")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR,
                        help="Directory to write the tokenizer_{1,2,3}.pkl files")
    parser.add_argument("--vocab_size", type=int, default=DEFAULT_VOCAB_SIZE,
                        help="Maximum vocabulary size for each tokenizer")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Random seed for reproducible training")
    parser.add_argument("--train_files_1", nargs="+", default=DEFAULT_TRAIN_FILES[1],
                        help="Training file name(s) for tokenizer 1 (under --data_dir)")
    parser.add_argument("--train_files_2", nargs="+", default=DEFAULT_TRAIN_FILES[2],
                        help="Training file name(s) for tokenizer 2 (under --data_dir)")
    parser.add_argument("--train_files_3", nargs="+", default=DEFAULT_TRAIN_FILES[3],
                        help="Training file name(s) for tokenizer 3 / hidden domain (under --data_dir)")
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
