from abc import ABC, abstractmethod
from typing import List, Dict, Union, Optional
import pickle
import os


class BaseTokenizer(ABC):
    """
    Base abstract class for tokenizers.
    Students will implement their own tokenizers derived from this class.
    """
    
    def __init__(self):
        """Initialize the tokenizer with empty vocabulary"""
        self.token_to_id = {}  # Token to ID mapping
        self.id_to_token = {}  # ID to token mapping
        self.special_tokens = {
            "[PAD]": 0,
            "[UNK]": 1,
            "[BOS]": 2,
            "[EOS]": 3,
        }
        
        # Initialize special tokens in the mappings
        for token, token_id in self.special_tokens.items():
            self.token_to_id[token] = token_id
            self.id_to_token[token_id] = token
    
    @abstractmethod
    def train(self, texts: List[str]) -> None:
        """
        Train the tokenizer on a list of texts
        
        Args:
            texts: List of training texts
        """
        pass
    
    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """
        Convert a text string into a list of token IDs
        
        Args:
            text: The input text to encode
            
        Returns:
            A list of token IDs
        """
        pass
    
    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """
        Convert a batch of text strings into token IDs
        
        Args:
            texts: List of input texts to encode
            
        Returns:
            A list of lists containing token IDs
        """
        return [self.encode(text) for text in texts]
    
    @abstractmethod
    def decode(self, token_ids: List[int]) -> str:
        """
        Convert a list of token IDs back to a text string
        
        Args:
            token_ids: List of token IDs to decode
            
        Returns:
            The decoded text string
        """
        pass
    
    def decode_batch(self, batch_token_ids: List[List[int]]) -> List[str]:
        """
        Convert a batch of token ID lists back to text strings
        
        Args:
            batch_token_ids: List of lists containing token IDs
            
        Returns:
            List of decoded text strings
        """
        return [self.decode(token_ids) for token_ids in batch_token_ids]
    
    def get_vocab_size(self) -> int:
        """
        Get the size of the vocabulary
        
        Returns:
            The number of tokens in the vocabulary
        """
        return len(self.token_to_id)
    
    def save(self, path: str) -> None:
        """
        Save the tokenizer to a file
        
        Args:
            path: Path where the tokenizer will be saved
        """
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: str) -> 'BaseTokenizer':
        """
        Load a tokenizer from a file
        
        Args:
            path: Path to the saved tokenizer
            
        Returns:
            The loaded tokenizer
        """
        with open(path, 'rb') as f:
            return pickle.load(f) 