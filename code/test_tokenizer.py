import argparse
import time
import os
from typing import List, Type
import sys
import pickle

from base_tokenizer import BaseTokenizer


def measure_encoding_speed(tokenizer: BaseTokenizer, texts: List[str], repeats: int = 5) -> float:
    """
    Measure tokenizer encoding speed in tokens per second
    Also returns total time and total tokens for the test set (no repeats)
    
    Args:
        tokenizer: Tokenizer to test
        texts: List of texts to encode
        repeats: Number of times to repeat the test
        
    Returns:
        Average tokens per second, total time for test set, total tokens for test set
    """
    total_tokens = 0
    start_time = time.time()
    
    for _ in range(repeats):
        for text in texts:
            tokens = tokenizer.encode(text)
            total_tokens += len(tokens)
    
    elapsed_time = time.time() - start_time
    tokens_per_second = total_tokens / elapsed_time
    
    # For reporting: single pass
    single_pass_tokens = 0
    single_pass_start = time.time()
    for text in texts:
        tokens = tokenizer.encode(text)
        single_pass_tokens += len(tokens)
    single_pass_time = time.time() - single_pass_start
    
    return tokens_per_second, single_pass_time, single_pass_tokens


def calculate_efficiency(tokenizer: BaseTokenizer, train_texts: List[str], test_texts: List[str]) -> float:
    """
    Calculate tokenizer efficiency (tokens per character) on test texts
    
    Args:
        tokenizer: Trained tokenizer
        train_texts: Texts used for training (for reference)
        test_texts: New texts to evaluate on
        
    Returns:
        Average number of tokens per character on test texts
    """
    total_chars = 0
    total_tokens = 0
    
    for text in test_texts:
        total_chars += len(text)
        tokens = tokenizer.encode(text)
        total_tokens += len(tokens)
    
    return total_tokens / total_chars if total_chars > 0 else 0


def test_reconstruction(tokenizer: BaseTokenizer, texts: List[str], sample_size: int = 10) -> float:
    """
    Test if tokenizer can correctly encode and decode texts
    
    Args:
        tokenizer: Tokenizer to test
        texts: List of texts to test on
        sample_size: Number of samples to test
        
    Returns:
        Percentage of texts that were correctly reconstructed
    """
    correct = 0
    test_texts = texts[:sample_size] if sample_size < len(texts) else texts
    
    for text in test_texts:
        token_ids = tokenizer.encode(text)
        reconstructed = tokenizer.decode(token_ids)
        
        # Check if original and reconstructed texts match
        # (allowing for whitespace differences)
        if reconstructed.replace(" ", "") == text.replace(" ", ""):
            correct += 1
    
    return correct / len(test_texts) if test_texts else 0


def load_texts(file_path: str) -> List[str]:
    """
    Load texts from a file
    
    Args:
        file_path: Path to the text file
        
    Returns:
        List of texts
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]


def main(tokenizer_path: str, domain_train_file: str, domain_test_file: str,
         repeats: int = 5, reconstruction_samples: int = 20):
    """
    Main function to test a tokenizer

    Args:
        tokenizer_path: Path to the trained tokenizer
        domain_train_file: Path to the domain training data
        domain_test_file: Path to the domain test data
        repeats: Number of passes for the encoding-speed measurement
        reconstruction_samples: Number of texts to test for reconstruction
    """
    print(f"Loading tokenizer from {tokenizer_path}")
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    
    # Load test texts
    print(f"Loading training texts from {domain_train_file}")
    train_texts = load_texts(domain_train_file)
    
    print(f"Loading test texts from {domain_test_file}")
    test_texts = load_texts(domain_test_file)
    
    # Test encoding speed
    print("\nTesting encoding speed...")
    tokens_per_second, total_time, total_tokens = measure_encoding_speed(tokenizer, test_texts, repeats=repeats)
    print(f"Encoding speed: {tokens_per_second:.2f} tokens/second")
    print(f"Total encoding time for test set: {total_time:.4f} seconds")
    print(f"Total tokens used for test set: {total_tokens}")
    
    # Test efficiency
    print("\nTesting tokenization efficiency...")
    tokens_per_char = calculate_efficiency(tokenizer, train_texts, test_texts)
    print(f"Tokens per character: {tokens_per_char:.4f}")
    
    # Test reconstruction
    print("\nTesting text reconstruction...")
    reconstruction_rate = test_reconstruction(tokenizer, test_texts, sample_size=reconstruction_samples)
    print(f"Reconstruction success rate: {reconstruction_rate * 100:.2f}%")
    
    print("\nSample encoding/decoding:")
    if test_texts:
        sample_text = test_texts[0]
        print(f"Original: {sample_text}")
        
        encoded = tokenizer.encode(sample_text)
        print(f"Encoded: {encoded[:50]}{'...' if len(encoded) > 50 else ''}")
        
        decoded = tokenizer.decode(encoded)
        print(f"Decoded: {decoded}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a trained tokenizer")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the trained tokenizer")
    parser.add_argument("--train_file", type=str, default="data/domain_1_train.txt", help="Domain training data file")
    parser.add_argument("--test_file", type=str, default=None, help="Domain test data file (defaults to train file if not specified)")
    parser.add_argument("--repeats", type=int, default=5, help="Number of passes for the encoding-speed measurement")
    parser.add_argument("--reconstruction_samples", type=int, default=20, help="Number of texts to test for reconstruction")

    args = parser.parse_args()

    # If test file is not specified, use the training file
    test_file = args.test_file if args.test_file else args.train_file

    main(args.tokenizer_path, args.train_file, test_file,
         repeats=args.repeats, reconstruction_samples=args.reconstruction_samples)
