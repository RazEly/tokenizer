import argparse
import os
import random
import numpy as np
from typing import List, Tuple, Dict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from base_tokenizer import BaseTokenizer
from tqdm import tqdm
import regex as re


# Fixed seed so every student's tokenizer is judged against an identically
# initialised and trained NER model (weight init, DataLoader shuffle, dropout).
# Do not change this value.
SEED = 42

# Fixed NER training hyperparameters — do not change. The grader runs an
# equivalent model uniformly for everyone; students may only change file paths.
OUTPUT_DIR = "models"
BATCH_SIZE = 32
LEARNING_RATE = 0.01
NUM_EPOCHS = 20


def set_seed(seed: int = SEED) -> None:
    """Seed all RNGs and force deterministic cuDNN for reproducible NER runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Dataset class for NER data
class NERDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[List[int]], tokenizer):
        self.tokenizer = tokenizer
        self.encoded_texts = []
        self.encoded_labels = []
        self.word_to_subtoken = []
        self.subtoken_to_word = []

        # Check if tokenizer has a space token
        self.space_token_id = None
        if hasattr(tokenizer, 'space_token') and tokenizer.space_token in tokenizer.token_to_id:
            self.space_token_id = tokenizer.token_to_id[tokenizer.space_token]

        # Process all texts with optimized alignment
        for text, text_labels in tqdm(zip(texts, labels), desc="Processing texts"):
            result = self._process_text_optimized(text, text_labels)
            if result:
                self.encoded_texts.append(result['token_ids'])
                self.encoded_labels.append(result['token_labels'])
                self.word_to_subtoken.append(result['word_to_sub'])
                self.subtoken_to_word.append(result['sub_to_word'])

    def _process_text_optimized(self, text: str, text_labels: List[int]) -> Dict:
        """Optimized version of text processing"""
        words = text.split()
        if not words:
            return None

        # Fast word span calculation using regex
        word_spans = self._get_word_spans_fast(text, words)

        # Encode text once
        token_ids = self.tokenizer.encode(text)

        # Fast token-to-character alignment
        token_char_spans = self._get_token_char_spans_fast(token_ids, text)

        # Fast alignment between tokens and words
        sub_to_word, word_to_sub = self._align_tokens_to_words_fast(
            token_char_spans, word_spans, len(words)
        )

        # Create token labels
        token_labels = self._create_token_labels(
            token_ids, word_to_sub, text_labels, sub_to_word
        )

        return {
            'token_ids': token_ids,
            'token_labels': token_labels,
            'word_to_sub': word_to_sub,
            'sub_to_word': sub_to_word
        }

    def _get_word_spans_fast(self, text: str, words: List[str]) -> List[Tuple[int, int]]:
        """Fast word span calculation using regex"""
        word_spans = []
        start = 0

        for word in words:
            # Find the word starting from the current position
            escaped_word = re.escape(word)
            match = re.search(escaped_word, text[start:])

            if match:
                word_start = start + match.start()
                word_end = word_start + len(word)
                word_spans.append((word_start, word_end))
                start = word_end
            else:
                # Fallback: approximate position
                word_start = start
                word_end = start + len(word)
                word_spans.append((word_start, word_end))
                start = word_end + 1  # +1 for space

        return word_spans

    def _get_token_char_spans_fast(self, token_ids: List[int], text: str) -> List[Tuple[int, int]]:
        """Fast token character span calculation"""
        if not token_ids:
            return []

        # Strategy 1: Use tokenizer's built-in alignment if available
        if hasattr(self.tokenizer, 'encode_with_offsets'):
            try:
                _, offsets = self.tokenizer.encode_with_offsets(text)
                return offsets
            except:
                pass

        # Strategy 2: Batch decode approach (much faster than incremental)
        token_char_spans = []
        decoded_so_far = ""

        for i, token_id in enumerate(token_ids):
            # Decode current token individually
            current_token_str = self.tokenizer.decode([token_id])

            # Find where this token should start in the original text
            expected_start = len(decoded_so_far)

            # Update decoded_so_far
            decoded_so_far = self.tokenizer.decode(token_ids[:i + 1])

            # Calculate span
            token_start = expected_start
            token_end = len(decoded_so_far)

            # Clamp to text bounds
            token_start = max(0, min(token_start, len(text)))
            token_end = max(token_start, min(token_end, len(text)))

            token_char_spans.append((token_start, token_end))

        return token_char_spans

    def _align_tokens_to_words_fast(self, token_char_spans: List[Tuple[int, int]],
                                    word_spans: List[Tuple[int, int]],
                                    num_words: int) -> Tuple[List[int], List[int]]:
        """Fast alignment using vectorized operations"""
        num_tokens = len(token_char_spans)

        # Initialize mappings
        sub_to_word = [-1] * num_tokens
        word_to_sub = [-1] * num_words

        # Create overlap matrix (optimized)
        for token_idx, (token_start, token_end) in enumerate(token_char_spans):
            best_word_idx = -1
            max_overlap = 0

            for word_idx, (word_start, word_end) in enumerate(word_spans):
                # Quick overlap calculation
                overlap = max(0, min(token_end, word_end) - max(token_start, word_start))

                if overlap > max_overlap:
                    max_overlap = overlap
                    best_word_idx = word_idx

            sub_to_word[token_idx] = best_word_idx

        # Find first token for each word
        for word_idx in range(num_words):
            for token_idx, assigned_word in enumerate(sub_to_word):
                if assigned_word == word_idx:
                    word_to_sub[word_idx] = token_idx
                    break

            # Fallback: find any overlapping token
            if word_to_sub[word_idx] == -1:
                word_start, word_end = word_spans[word_idx]
                for token_idx, (token_start, token_end) in enumerate(token_char_spans):
                    if max(token_start, word_start) < min(token_end, word_end):
                        word_to_sub[word_idx] = token_idx
                        break

                # Last resort
                if word_to_sub[word_idx] == -1:
                    word_to_sub[word_idx] = min(word_idx, num_tokens - 1)

        return sub_to_word, word_to_sub

    def _create_token_labels(self, token_ids: List[int], word_to_sub: List[int],
                             text_labels: List[int], sub_to_word: List[int]) -> List[int]:
        """Create token labels efficiently"""
        token_labels = [-100] * len(token_ids)

        # Handle space tokens
        if self.space_token_id is not None:
            for token_idx, token_id in enumerate(token_ids):
                if token_id == self.space_token_id:
                    # Don't change sub_to_word here to avoid breaking alignment
                    pass

        # Assign labels to first token of each word
        for word_idx, first_token_idx in enumerate(word_to_sub):
            if 0 <= first_token_idx < len(token_labels) and word_idx < len(text_labels):
                token_labels[first_token_idx] = text_labels[word_idx]

        return token_labels

    def __len__(self):
        return len(self.encoded_texts)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.encoded_texts[idx], dtype=torch.long),
            'labels': torch.tensor(self.encoded_labels[idx], dtype=torch.long),
            'word_to_subtoken': self.word_to_subtoken[idx],
            'subtoken_to_word': self.subtoken_to_word[idx],
        }
# Simple NER model
class NERModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden_dim: int = 256, num_classes: int = 2):
        super(NERModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)  # *2 for bidirectional

    def forward(self, input_ids):
        # Get embeddings
        embeddings = self.embeddings(input_ids)

        # Pass through LSTM
        lstm_output, _ = self.lstm(embeddings)

        # Apply dropout
        lstm_output = self.dropout(lstm_output)

        # Classify each token
        logits = self.classifier(lstm_output)

        return logits


def pad_sequences(batch, pad_id: int = 0):
    """
    Pad sequences in a batch to the same length

    Args:
        batch: Batch of sequences
        pad_id: ID to use for padding

    Returns:
        Padded batch and mask
    """
    max_len = max(len(seq) for seq in batch)
    padded_batch = []
    mask = []

    for seq in batch:
        padded_seq = seq + [pad_id] * (max_len - len(seq))
        padded_batch.append(padded_seq)
        mask.append([1] * len(seq) + [0] * (max_len - len(seq)))

    return padded_batch, mask


def collate_fn(batch):
    """
    Collate function for DataLoader

    Args:
        batch: Batch of samples

    Returns:
        Collated batch with padded sequences
    """
    input_ids = [item['input_ids'].tolist() for item in batch]
    labels = [item['labels'].tolist() for item in batch]
    word_to_subtoken = [item['word_to_subtoken'] for item in batch]
    subtoken_to_word = [item['subtoken_to_word'] for item in batch]
    input_ids, attention_mask = pad_sequences(input_ids, pad_id=0)
    labels, _ = pad_sequences(labels, pad_id=-100)
    # No padding for word_to_subtoken/subtoken_to_word (used only for evaluation, not for loss)
    return {
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
        'attention_mask': torch.tensor(attention_mask, dtype=torch.float),
        'labels': torch.tensor(labels, dtype=torch.long),
        'word_to_subtoken': word_to_subtoken,
        'subtoken_to_word': subtoken_to_word,
    }


def read_ner_data(file_path: str) -> Tuple[List[str], List[List[int]]]:
    """
    Read NER data from a file

    Args:
        file_path: Path to the NER data file

    Returns:
        Tuple of (texts, labels)
    """
    texts = []
    labels = []

    current_text = []
    current_labels = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            # Empty line indicates end of sentence
            if not line:
                if current_text:
                    texts.append(' '.join(current_text))
                    labels.append(current_labels)
                    current_text = []
                    current_labels = []
                continue

            # Parse token and tag
            parts = line.split('\t')
            if len(parts) == 2:
                token, tag = parts
                # Convert tag to binary (1 if entity, 0 if not)
                label = 1 if tag != '0' else 0
                current_text.append(token)
                current_labels.append(label)

    # Add the last sentence if it exists
    if current_text:
        texts.append(' '.join(current_text))
        labels.append(current_labels)

    return texts, labels


def evaluate_model(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Dict:
    """
    Evaluate the model on a dataset

    Args:
        model: The NER model
        dataloader: DataLoader for evaluation
        device: Device to run evaluation on

    Returns:
        Evaluation metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            word_to_subtoken = batch['word_to_subtoken']
            # Forward pass
            logits = model(input_ids)
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            labels_np = labels.numpy()

            # For each sentence in batch
            for i in range(len(preds)):
                # For each word, get prediction from its first subtoken
                for word_idx, subtoken_idx in enumerate(word_to_subtoken[i]):
                    # Skip if subtoken_idx is out of bounds (due to padding)
                    if subtoken_idx >= len(preds[i]):
                        continue

                    pred_label = preds[i][subtoken_idx]

                    # For evaluation, we compare each word's prediction against its gold label
                    # We get the gold label for this word (from its first subtoken)
                    gold_label = labels_np[i][subtoken_idx]

                    # Only consider valid labels (skip -100)
                    if gold_label == -100:
                        continue

                    all_preds.append(pred_label)
                    all_labels.append(gold_label)

    # Calculate metrics
    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    total = len(all_preds)
    accuracy = correct / total if total > 0 else 0

    true_positives = sum(p == 1 and l == 1 for p, l in zip(all_preds, all_labels))
    pred_positives = sum(p == 1 for p in all_preds)
    actual_positives = sum(l == 1 for l in all_labels)

    precision = true_positives / pred_positives if pred_positives > 0 else 0
    recall = true_positives / actual_positives if actual_positives > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def train_ner_model(
    tokenizer_path: str,
    train_file: str,
    dev_file: str,
    output_dir: str,
    batch_size: int = 32,
    lr: float = 0.001,
    num_epochs: int = 5
):
    """
    Train a NER model

    Args:
        tokenizer_path: Path to the trained tokenizer
        train_file: Path to the training data
        dev_file: Path to the development data
        output_dir: Directory to save the trained model
        batch_size: Batch size for training
        lr: Learning rate
        num_epochs: Number of training epochs
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Seed everything before any model/data randomness for reproducibility
    set_seed()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load tokenizer
    print(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = BaseTokenizer.load(tokenizer_path)

    # Load data
    print(f"Loading training data from {train_file}")
    train_texts, train_labels = read_ner_data(train_file)

    print(f"Loading dev data from {dev_file}")
    dev_texts, dev_labels = read_ner_data(dev_file)

    # Create datasets
    print("Creating datasets")
    train_dataset = NERDataset(train_texts, train_labels, tokenizer)
    dev_dataset = NERDataset(dev_texts, dev_labels, tokenizer)

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, collate_fn=collate_fn)

    # Initialize model
    print("Initializing model")
    model = NERModel(tokenizer.get_vocab_size(), num_classes=2)  # Binary classification
    model.to(device)

    # Initialize optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)  # Ignore padded tokens

    # Training loop
    print(f"Starting training for {num_epochs} epochs")
    best_f1 = 0.0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            logits = model(input_ids)

            # Reshape for loss calculation
            B, T, C = logits.shape  # Batch size, sequence length, num classes
            loss = loss_fn(logits.view(-1, C), labels.view(-1))

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Evaluate on dev set
        metrics = evaluate_model(model, dev_dataloader, device)

        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train loss: {total_loss / len(train_dataloader):.4f}")
        print(f"  Dev accuracy: {metrics['accuracy']:.4f}")
        print(f"  Dev precision: {metrics['precision']:.4f}")
        print(f"  Dev recall: {metrics['recall']:.4f}")
        print(f"  Dev F1: {metrics['f1']:.4f}")

        # Save best model
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            model_path = os.path.join(output_dir, "best_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'f1': best_f1
            }, model_path)
            print(f"  Saved new best model with F1: {best_f1:.4f}")

    print(f"Training complete. Best F1: {best_f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a NER model")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the trained tokenizer")
    parser.add_argument("--train_file", type=str, default="data/ner_data/train_1_binary.tagged", help="Path to the training data")
    parser.add_argument("--dev_file", type=str, default="data/ner_data/dev_1_binary.tagged", help="Path to the development data")
    args = parser.parse_args()
    # Only the file paths above may change; the model hyperparameters below are
    # fixed so every tokenizer is judged against an identical NER model.
    train_ner_model(
        args.tokenizer_path,
        args.train_file,
        args.dev_file,
        OUTPUT_DIR,
        BATCH_SIZE,
        LEARNING_RATE,
        NUM_EPOCHS,
    )