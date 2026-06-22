import argparse
import os
from typing import List

from bpe_tokenizer import BPETokenizer


def read_text_file(file_path: str) -> List[str]:
    """
    Read lines from a text file
    
    Args:
        file_path: Path to the text file
        
    Returns:
        List of lines from the file
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.readlines()


def train_tokenizer(domain_file: str, output_dir: str, vocab_size: int = 5000):
    """
    Train a tokenizer on domain data and save it
    
    Args:
        domain_file: Path to the domain training data file
        output_dir: Directory where to save the trained tokenizer
        vocab_size: Maximum vocabulary size
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read domain data
    print(f"Reading domain data from {domain_file}")
    texts = read_text_file(domain_file)
    print(f"Read {len(texts)} lines of text")

    # Initialize and train tokenizer
    print(f"Training BPE tokenizer with vocab size {vocab_size}")
    tokenizer = BPETokenizer(vocab_size=vocab_size)
    tokenizer.train(texts)
    
    # Save the tokenizer
    output_path = os.path.join(output_dir, "tokenizer.pkl")
    print(f"Saving tokenizer to {output_path}")
    tokenizer.save(output_path)
    print(f"Tokenizer trained with {tokenizer.get_vocab_size()} tokens")
    
    # Test the tokenizer on a sample
    if texts:
        sample_text = texts[0].strip()
        print("\nExample encoding/decoding:")
        print(f"Original text: {sample_text}")
        
        encoded = tokenizer.encode(sample_text)
        print(f"Encoded: {encoded[:50]}{'...' if len(encoded) > 50 else ''}")
        
        decoded = tokenizer.decode(encoded)
        print(f"Decoded: {decoded}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer on domain data")
    parser.add_argument("--domain_file", type=str, required=True, help="Path to the domain data file")
    parser.add_argument("--output_dir", type=str, default="tokenizers", help="Directory to save the tokenizer")
    parser.add_argument("--vocab_size", type=int, default=5000, help="Maximum vocabulary size")
    
    args = parser.parse_args()
    
    train_tokenizer(args.domain_file, args.output_dir, args.vocab_size) 