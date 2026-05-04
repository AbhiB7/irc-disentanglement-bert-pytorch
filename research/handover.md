# IRC Conversation Disentanglement - Research Handover

## Project Overview

This project implements a BERT-based CrossEncoder model for IRC conversation disentanglement. The model predicts whether a message is a reply to another message in IRC conversations.

## Source Files

### 1. src/data_loader.py

```python
"""
IRC Conversation Disentanglement Data Loader for PyTorch BERT

This module loads the IRC dataset and creates pairs of messages for training/evaluation.
Matches the original data format from archive/jkummerfield-original/src/disentangle.py
but returns PyTorch Dataset compatible format.

Key differences from original:
- Returns (text_pair, label, features) instead of raw instances
- Uses only 4 handcrafted features as per plan (not 77)
- Compatible with BERT tokenization
"""

import os
import re
import logging
from typing import List, Tuple, Dict, Set, Optional
from dataclasses import dataclass
from datetime import datetime

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm

# Get logger for this module - logging should be configured by the main entry point
logger = logging.getLogger(__name__)


@dataclass
class IRCMessage:
    """Represents a single IRC message with metadata"""

    index: int
    timestamp: Optional[Tuple[int, int]]  # (hour, minute) or None for system messages
    speaker: str
    text: str
    is_system: bool
    is_bot: bool
    targets: Set[str]  # users mentioned/targeted in this message
    last_from_same_user: Optional[int]  # index of previous message from same user
    next_from_same_user: Optional[int]  # index of next message from same user


@dataclass
class IRCConversation:
    """Represents a single IRC log file with messages and gold links"""

    name: str
    messages: List[IRCMessage]
    gold_links: Dict[int, List[int]]  # child -> list of parent indices
    # Additional metadata for feature extraction
    user_message_indices: Dict[str, List[int]]  # user -> list of message indices


def parse_irc_line(line: str) -> Tuple[Optional[Tuple[int, int]], str, str, bool]:
    """
    Parse an IRC line in format: [HH:MM] <Speaker> message
    Returns: (timestamp, speaker, text, is_system)
    """
    line = line.strip()

    # System messages start with "==="
    if line.startswith("==="):
        return None, "SYSTEM", line, True

    # Regular message format: [HH:MM] <Speaker> message
    match = re.match(r"^\[(\d{2}):(\d{2})\] <([^>]+)> (.*)$", line)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2))
        speaker = match.group(3)
        text = match.group(4)
        return (hour, minute), speaker, text, False

    # Some messages might have different format, fallback
    return None, "UNKNOWN", line, True


def extract_targets(text: str, users: Set[str]) -> Set[str]:
    """
    Extract target users from message text.
    Simplified version of original get_targets function.
    """
    targets = set()
    text_lower = text.lower()

    # Check for direct mentions (common IRC patterns)
    for user in users:
        user_lower = user.lower()
        # Simple check: user mentioned as word boundary
        if re.search(r"\b" + re.escape(user_lower) + r"\b", text_lower):
            targets.add(user)

    return targets


def load_conversation(ascii_path: str, annotation_path: str) -> IRCConversation:
    """
    Load a conversation from ASCII and annotation files.
    Matches the original read_data function logic.
    """
    logger.info(f"Loading conversation from {ascii_path}")
    start_time = datetime.now()

    # Read ASCII file
    with open(ascii_path, "r", encoding="utf-8") as f:
        ascii_lines = [line.rstrip("\n") for line in f]

    logger.info(f"  Read {len(ascii_lines)} lines from ASCII file")

    # Parse all messages
    messages = []
    users = set()

    for idx, line in enumerate(ascii_lines):
        timestamp, speaker, text, is_system = parse_irc_line(line)

        # Update users (non-system, non-bot)
        if not is_system and speaker not in ["SYSTEM", "UNKNOWN"]:
            users.add(speaker)

        messages.append(
            IRCMessage(
                index=idx,
                timestamp=timestamp,
                speaker=speaker,
                text=text,
                is_system=is_system,
                is_bot=(speaker in ["ubottu", "ubotu"]),
                targets=set(),  # Will be populated after we have users
                last_from_same_user=None,
                next_from_same_user=None,
            )
        )

    logger.info(f"  Parsed {len(messages)} messages, found {len(users)} unique users")

    # Now extract targets for each message
    for msg in messages:
        if not msg.is_system:
            msg.targets = extract_targets(msg.text, users)

    # Build user message indices
    user_message_indices = {}
    for idx, msg in enumerate(messages):
        if not msg.is_system:
            user_message_indices.setdefault(msg.speaker, []).append(idx)

    # Set last_from_same_user and next_from_same_user
    for user, indices in user_message_indices.items():
        for i, idx in enumerate(indices):
            messages[idx].last_from_same_user = indices[i - 1] if i > 0 else None
            messages[idx].next_from_same_user = (
                indices[i + 1] if i < len(indices) - 1 else None
            )

    # Load gold links from annotation file
    gold_links = {}
    with open(annotation_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0] != "-":
                child = int(parts[0])
                parent = int(parts[1])
                gold_links.setdefault(child, []).append(parent)

    logger.info(f"  Loaded {len(gold_links)} gold links from annotation file")

    # Get base name without extensions
    name = os.path.basename(ascii_path)
    for ext in [".ascii.txt", ".annotation.txt", ".raw.txt", ".tok.txt"]:
        if name.endswith(ext):
            name = name[: -len(ext)]

    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"  Conversation '{name}' loaded in {elapsed:.2f}s")

    return IRCConversation(
        name=name,
        messages=messages,
        gold_links=gold_links,
        user_message_indices=user_message_indices,
    )


def compute_features(
    msg_i: IRCMessage,
    msg_j: IRCMessage,
    conversation: IRCConversation,
    max_dist: int = 30,
) -> List[float]:
    """
    Compute 4 handcrafted features as per project plan:
    1. time_diff_min: Time difference in minutes (capped at 100)
    2. speaker_match: 1 if same speaker, 0 otherwise
    3. pos_dist: Position distance (j - i) normalized by max_dist
    4. word_jaccard: Jaccard similarity of word sets

    Returns: List of 4 feature values
    """
    MAX_DIST = max_dist  # Use provided max_dist for normalization

    # 1. Time difference in minutes
    time_diff = 0.0
    if msg_i.timestamp and msg_j.timestamp:
        hi, mi = msg_i.timestamp
        hj, mj = msg_j.timestamp
        if hi == hj:
            time_diff = abs(mj - mi)
        else:
            time_diff = abs((hj * 60 + mj) - (hi * 60 + mi))
    time_diff_norm = min(time_diff / 60.0, 1.0)  # Normalize to 0-1, cap at 60min

    # 2. Speaker match
    speaker_match = 1.0 if msg_i.speaker == msg_j.speaker else 0.0

    # 3. Position distance (normalized)
    pos_dist = abs(msg_j.index - msg_i.index)
    pos_dist_norm = min(pos_dist / MAX_DIST, 1.0)

    # 4. Word Jaccard similarity
    words_i = set(msg_i.text.lower().split())
    words_j = set(msg_j.text.lower().split())
    if len(words_i) == 0 or len(words_j) == 0:
        jaccard = 0.0
    else:
        intersection = len(words_i.intersection(words_j))
        union = len(words_i.union(words_j))
        jaccard = intersection / union if union > 0 else 0.0

    return [time_diff_norm, speaker_match, pos_dist_norm, jaccard]


class IRCDisentanglementDataset(Dataset):
    """
    PyTorch Dataset for IRC conversation disentanglement.
    Creates message pairs with labels and features.
    """

    def __init__(
        self,
        ascii_files: List[str],
        annotation_files: List[str],
        tokenizer,
        max_dist: int = 30,
        max_length: int = 128,
        skip_labels: bool = False,
        test_start: int = 0,
        test_end: int = 1000000000, # Default to 1 Billion (effectively no limit).
                                    # Note: Previous limit of 1M was too low for full dataset.
    ):
        """
        Args:
            ascii_files: List of ASCII file paths
            annotation_files: List of annotation file paths (parallel to ascii_files)
            tokenizer: BERT tokenizer
            max_dist: Maximum distance to consider for linking (default 30)
            max_length: Maximum token length for BERT
            skip_labels: If True, do not use gold labels (for blind test)
            test_start/end: Which messages to process in each file
        """
        assert len(ascii_files) == len(annotation_files), "File lists must match"

        self.tokenizer = tokenizer
        self.max_dist = max_dist
        self.max_length = max_length
        self.skip_labels = skip_labels
        self.test_start = test_start
        self.test_end = test_end

        # Load all conversations
        self.conversations = []
        self.conversation_map = (
            []
        )  # Maps pair index to (conv_idx, msg_i_idx, msg_j_idx)
        self.pairs = []  # List of (text_pair, label, features) - raw text for lazy tokenization

        logger.info(
            f"Initializing IRCDisentanglementDataset with {len(ascii_files)} files"
        )
        logger.info(
            f"  max_dist={max_dist}, max_length={max_length}, skip_labels={skip_labels}"
        )
        logger.info(f"  test_start={test_start}, test_end={test_end}")

        start_time = datetime.now()

        for idx, (ascii_path, ann_path) in enumerate(
            tqdm(
                zip(ascii_files, annotation_files),
                total=len(ascii_files),
                desc="Loading conversations",
                leave=True,
            )
        ):
            conv = load_conversation(ascii_path, ann_path)
            self.conversations.append(conv)

            # Create message pairs
            self._create_pairs_for_conversation(conv, len(self.conversations) - 1)

            logger.info(
                f"  File {idx+1}/{len(ascii_files)}: {conv.name} - {len(conv.messages)} messages, {len(self.pairs)} total pairs so far"
            )

            # Early exit if we've reached test_end pairs (if limiting)
            # Check against test_end limit (now supports values > 1M)
            if self.test_end < 1000000000 and len(self.pairs) >= self.test_end:
                logger.info(
                    f"  Reached test_end limit ({self.test_end} pairs), stopping early"
                )
                break

        # Truncate pairs to test_end if specified
        if self.test_end < 1000000000 and self.test_end < len(self.pairs):
            logger.info(
                f"Truncating pairs from {len(self.pairs)} to {self.test_end} (test_end limit)"
            )
            self.pairs = self.pairs[: self.test_end]
            self.conversation_map = self.conversation_map[: self.test_end]

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(
            f"Dataset initialization complete: {len(self.pairs)} pairs from {len(self.conversations)} conversations in {elapsed:.2f}s"
        )

    def _create_pairs_for_conversation(self, conv: IRCConversation, conv_idx: int):
        """Create all message pairs for a conversation"""
        messages = conv.messages
        gold_links = conv.gold_links

        # Determine which messages to process
        # If test_end is small, we limit messages per file to speed up loading
        # If limiting data, only process a subset of messages per file to speed up loading
        if self.test_end < 1000000000:
            start_idx = self.test_start
            end_idx = min(self.test_end, len(messages))
            process_indices = range(start_idx, end_idx)
            logger.info(
                f"  Creating pairs for {conv.name}: messages {start_idx} to {end_idx} (limited)"
            )
        else:
            process_indices = range(len(messages))
            logger.info(
                f"  Creating pairs for {conv.name}: all {len(messages)} messages"
            )

        pairs_before = len(self.pairs)

        # Use tqdm for progress bar on message iteration
        for i in tqdm(
            process_indices,
            desc=f"  Pairs for {conv.name}",
            leave=False,
            disable=len(process_indices) < 100,
        ):
            msg_i = messages[i]

            # For each possible parent within max_dist
            for j in range(max(0, i - self.max_dist + 1), i + 1):
                msg_j = messages[j]

                # Skip system messages as parents (except self-links)
                if j != i and msg_j.is_system:
                    continue

                # Create pair - store RAW text for lazy tokenization
                text_pair = [msg_j.text, msg_i.text]  # [parent, child]

                # Label: 1 if j is a gold parent of i, 0 otherwise
                label = 1.0 if (i in gold_links and j in gold_links[i]) else 0.0

                # For blind test mode, we don't have gold labels
                if self.skip_labels:
                    label = -1.0  # Placeholder

                # Compute features
                features = compute_features(
                    msg_j, msg_i, conv, max_dist=self.max_dist
                )  # parent, child

                # Store raw text pair - tokenization happens in __getitem__ (lazy)
                self.pairs.append((text_pair, label, features))
                self.conversation_map.append((conv_idx, i, j))

        pairs_added = len(self.pairs) - pairs_before
        logger.info(f"  Created {pairs_added} pairs for {conv.name}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        text_pair, label, features = self.pairs[idx]

        # Tokenize on-the-fly (lazy tokenization to save RAM)
        encoding = self.tokenizer(
            text_pair[0],  # parent text
            text_pair[1],  # child text
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "features": torch.tensor(features, dtype=torch.float32),
            "labels": (
                torch.tensor(label, dtype=torch.float32)
                if label != -1
                else torch.tensor(0.0, dtype=torch.float32)
            ),
        }

        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"].squeeze(0)

        return item


def load_dataset_files(
    data_dir: str, split: str = "train"
) -> Tuple[List[str], List[str]]:
    """
    Load file paths for a given split.
    Returns: (ascii_files, annotation_files)
    """
    import glob

    if split == "train":
        pattern = os.path.join(data_dir, "train", "*.ascii.txt")
    elif split == "dev":
        pattern = os.path.join(data_dir, "dev", "*.ascii.txt")
    elif split == "test":
        pattern = os.path.join(data_dir, "test", "*.ascii.txt")
    else:
        raise ValueError(f"Unknown split: {split}")

    ascii_files = sorted(glob.glob(pattern))
    annotation_files = []

    for ascii_file in ascii_files:
        ann_file = ascii_file.replace(".ascii.txt", ".annotation.txt")
        if os.path.exists(ann_file):
            annotation_files.append(ann_file)
        else:
            print(f"Warning: No annotation file for {ascii_file}")

    return ascii_files, annotation_files


if __name__ == "__main__":
    # Data loader module - use train.py for training
    pass
```

---

### 2. src/model.py

```python
"""
IRC Conversation Disentanglement Model - BERT CrossEncoder with Handcrafted Features

Architecture:
1. BERT CrossEncoder processes message pairs
2. Extract [CLS] token embedding (768-dim)
3. Concatenate with 4 handcrafted features → 772-dim vector
4. Linear layer (772 → 1) + Sigmoid for binary classification

Matches the architecture described in context/CONTEXT.md
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from typing import Optional, Tuple, Dict


class CrossEncoderWithFeatures(nn.Module):
    """
    BERT-based CrossEncoder with additional handcrafted features.
    
    Input: 
    - Tokenized message pairs (input_ids, attention_mask, token_type_ids)
    - 4 handcrafted features: [time_diff, speaker_match, pos_dist, word_jaccard]
    
    Output:
    - Probability that message_j is a reply to message_i (0-1)
    """
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_features: int = 4,
        dropout: float = 0.1,
        freeze_bert: bool = False
    ):
        super().__init__()
        
        # Load BERT model for CrossEncoder
        self.bert = AutoModel.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        
        # Freeze BERT layers if requested
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # BERT hidden size (typically 768 for bert-base-uncased)
        bert_hidden_size = self.config.hidden_size
        
        # Combined feature size
        combined_size = bert_hidden_size + num_features
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(combined_size, 1)
        
        # Initialize classifier weights
        self._init_weights(self.classifier)
        
        # Store dimensions for reference
        self.bert_hidden_size = bert_hidden_size
        self.num_features = num_features
        self.combined_size = combined_size
    
    def _init_weights(self, module):
        """Initialize weights for linear layers"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        features: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            token_type_ids: Token type IDs (segment IDs) [batch_size, seq_len]
            features: Handcrafted features [batch_size, num_features]
            labels: Ground truth labels [batch_size]
            
        Returns:
            Dictionary with:
            - logits: Raw model outputs [batch_size]
            - probs: Sigmoid probabilities [batch_size]
            - loss: BCE loss (if labels provided)
        """
        batch_size = input_ids.shape[0]
        
        # Get BERT embeddings
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        
        # Use [CLS] token embedding (first token)
        cls_embedding = bert_outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        
        # Apply dropout
        cls_embedding = self.dropout(cls_embedding)
        
        # Concatenate with handcrafted features
        if features is not None:
            # Ensure features have correct shape
            if features.dim() == 1:
                features = features.unsqueeze(0)
            
            # Verify feature dimension
            if features.shape[-1] != self.num_features:
                raise ValueError(
                    f"Expected {self.num_features} features, got {features.shape[-1]}"
                )
            
            # Concatenate BERT embedding with features
            combined = torch.cat([cls_embedding, features], dim=-1)  # [batch_size, hidden_size + num_features]
        else:
            # If no features provided, use zero-padded features
            zero_features = torch.zeros(
                batch_size, self.num_features,
                device=cls_embedding.device,
                dtype=cls_embedding.dtype
            )
            combined = torch.cat([cls_embedding, zero_features], dim=-1)
        
        # Classification head
        logits = self.classifier(combined).squeeze(-1)  # [batch_size]
        probs = torch.sigmoid(logits)
        
        # Prepare output
        outputs = {
            'logits': logits,
            'probs': probs
        }
        
        # Compute loss if labels provided
        if labels is not None:
            # Dynamic pos_weight based on actual batch label distribution
            # Clamp prevents explosion on batches with zero positives
            # Cap raised to 1500 to handle ~746:1 imbalance (was 300, insufficient)
            num_neg = (labels == 0).sum().float()
            num_pos = (labels == 1).sum().float()
            pos_weight = (num_neg / (num_pos + 1e-8)).clamp(min=10.0, max=1500.0)
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(logits.device))
            loss = loss_fn(logits, labels)
            outputs['loss'] = loss
        
        return outputs
    
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        features: Optional[torch.Tensor] = None,
        threshold: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with thresholding.
        
        Returns:
            - predictions: Binary predictions (0 or 1) [batch_size]
            - probabilities: Sigmoid probabilities [batch_size]
        """
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                features=features
            )
            
            probs = outputs['probs']
            predictions = (probs >= threshold).long()
            
            return predictions, probs


def create_model(
    model_name: str = "bert-base-uncased",
    num_features: int = 4,
    dropout: float = 0.1,
    freeze_bert: bool = False,
    device: str = None
) -> CrossEncoderWithFeatures:
    """
    Factory function to create and initialize model.
    
    Args:
        model_name: Pretrained BERT model name
        num_features: Number of handcrafted features
        dropout: Dropout probability
        freeze_bert: Whether to freeze BERT parameters
        device: Device to load model on (cuda/cpu)
        
    Returns:
        Initialized CrossEncoderWithFeatures model
    """
    model = CrossEncoderWithFeatures(
        model_name=model_name,
        num_features=num_features,
        dropout=dropout,
        freeze_bert=freeze_bert
    )
    
    # Move to device if specified
    if device:
        model = model.to(device)
    
    return model


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count trainable and total parameters.
    
    Returns:
        (trainable_params, total_params)
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    return trainable_params, total_params


# Test function to verify model works
def test_model():
    """Test the model with dummy data"""
    print("Testing CrossEncoderWithFeatures model...")
    
    # Create model
    model = create_model(
        model_name="bert-base-uncased",
        num_features=4,
        dropout=0.1,
        freeze_bert=False
    )
    
    # Count parameters
    trainable, total = count_parameters(model)
    print(f"  Parameters: {trainable:,} trainable, {total:,} total")
    print(f"  BERT hidden size: {model.bert_hidden_size}")
    print(f"  Combined size: {model.combined_size}")
    
    # Create dummy batch
    batch_size = 2
    seq_len = 128
    
    # Random inputs (simulating tokenized message pairs)
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones((batch_size, seq_len))
    token_type_ids = torch.zeros((batch_size, seq_len), dtype=torch.long)
    features = torch.randn((batch_size, 4))
    labels = torch.tensor([1.0, 0.0])
    
    # Forward pass
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        features=features,
        labels=labels
    )
    
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Features shape: {features.shape}")
    print(f"  Logits shape: {outputs['logits'].shape}")
    print(f"  Probs shape: {outputs['probs'].shape}")
    print(f"  Loss: {outputs.get('loss', 'N/A')}")
    
    # Test prediction
    predictions, probs = model.predict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        features=features,
        threshold=0.5
    )
    
    print(f"  Predictions: {predictions}")
    print(f"  Probabilities: {probs}")
    
    # Verify architecture
    assert outputs['logits'].shape == (batch_size,), "Logits should have shape [batch_size]"
    assert outputs['probs'].shape == (batch_size,), "Probs should have shape [batch_size]"
    assert model.combined_size == 768 + 4, f"Combined size should be 772, got {model.combined_size}"
    
    print("\n✓ Model test passed!")
    return model


if __name__ == "__main__":
    # Run test
    model = test_model()
```

---

### 3. src/train.py

```python
"""
IRC Conversation Disentanglement Training Script

Trains a BERT-based CrossEncoder with handcrafted features for IRC message linking.
Uses the tested data_loader.py and model.py modules.
"""

import argparse
import os
import sys
import time
import json
import logging
import psutil
import platform
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import IRCDisentanglementDataset, load_dataset_files
from model import CrossEncoderWithFeatures, create_model, count_parameters

# Configure logging
# Honor LOG_DIR environment variable if set, otherwise default to repo-local logs/
LOG_DIR_PATH = os.environ.get("LOG_DIR")
if LOG_DIR_PATH:
    LOG_DIR = Path(LOG_DIR_PATH)
else:
    LOG_DIR = Path(__file__).parent.parent / "logs"

LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train IRC Conversation Disentanglement Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "dev-only", "test"],
        default="train",
        help="Training mode: train (full), dev-only (single dev file), test (evaluate only)",
    )

    # Data paths
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing train/dev/test subdirectories",
    )

    # Model configuration
    parser.add_argument(
        "--model-name",
        type=str,
        default="bert-base-uncased",
        help="Pretrained BERT model name",
    )

    parser.add_argument(
        "--max-length", type=int, default=128, help="Maximum token length for BERT"
    )

    parser.add_argument(
        "--max-dist",
        type=int,
        default=30,
        help="Maximum distance to consider for linking (Reduced from 101 for local GPU feasibility)",
    )

    # Training hyperparameters
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for training (Optimized for RTX 5070 12GB)"
    )

    parser.add_argument(
        "--learning-rate", type=float, default=5e-5, help="Learning rate for optimizer"
    )

    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs"
    )

    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=100,
        help="Number of warmup steps for scheduler",
    )

    parser.add_argument(
        "--dropout", type=float, default=0.1, help="Dropout probability"
    )

    parser.add_argument(
        "--freeze-bert",
        action="store_true",
        help="Freeze BERT parameters during training",
    )

    # Checkpointing
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Number of epochs to wait for improvement before early stopping (0 to disable)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints and results",
    )

    parser.add_argument(
        "--save-every", type=int, default=1, help="Save checkpoint every N epochs"
    )

    parser.add_argument(
        "--eval-every", type=int, default=1, help="Evaluate on dev set every N epochs"
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of worker threads for data loading",
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use mixed precision training (FP16)",
    )

    # Threshold for prediction
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Threshold for binary prediction"
    )

    # Test mode options
    parser.add_argument(
        "--test-start",
        type=int,
        default=0,
        help="Start index for test mode (for limiting pairs)",
    )

    parser.add_argument(
        "--test-end",
        type=int,
        default=1000000,
        help="End index for test mode (for limiting pairs)",
    )

    # Resume training
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )

    return parser.parse_args()


def create_dataloaders(args, tokenizer):
    """Create train and dev dataloaders"""

    if args.mode == "dev-only":
        # Use only dev set
        try:
            dev_ascii, dev_ann = load_dataset_files(args.data_dir, "dev")
        except Exception as e:
            logger.error(f"Failed to load dev dataset files from {args.data_dir}: {e}")
            raise e

        if not dev_ascii:
            raise ValueError(f"No dev files found in {args.data_dir}")

        # Use first dev file only
        dev_ascii = [dev_ascii[0]]
        dev_ann = [dev_ann[0]]

        logger.info(f"Creating dev-only dataloader with 1 file: {dev_ascii[0]}")
        logger.info(f"  test_start={args.test_start}, test_end={args.test_end}")

        # Determine if we should skip labels (blind test)
        skip_labels = args.mode == "test"

        dev_dataset = IRCDisentanglementDataset(
            ascii_files=dev_ascii,
            annotation_files=dev_ann,
            tokenizer=tokenizer,
            max_dist=args.max_dist,
            max_length=args.max_length,
            skip_labels=skip_labels,
            test_start=args.test_start,
            test_end=args.test_end,
        )

        logger.info(f"  Dev dataset created: {len(dev_dataset)} pairs")

        dev_loader = DataLoader(
            dev_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(args.device == "cuda"),
        )

        return None, dev_loader

    else:
        # Load train and dev sets
        try:
            train_ascii, train_ann = load_dataset_files(args.data_dir, "train")
            dev_ascii, dev_ann = load_dataset_files(args.data_dir, "dev")
        except Exception as e:
            logger.error(f"Failed to load dataset files from {args.data_dir}: {e}")
            raise e

        if not train_ascii:
            raise ValueError(f"No train files found in {args.data_dir}")
        if not dev_ascii:
            raise ValueError(f"No dev files found in {args.data_dir}")

        logger.info(f"Loading {len(train_ascii)} train files...")
        # For Test 1, we want to limit the training set size as well
        # But we never skip labels during training
        train_dataset = IRCDisentanglementDataset(
            ascii_files=train_ascii,
            annotation_files=train_ann,
            tokenizer=tokenizer,
            max_dist=args.max_dist,
            max_length=args.max_length,
            skip_labels=False,
            test_start=args.test_start,
            test_end=args.test_end,
        )

        logger.info(f"  Train dataset created: {len(train_dataset)} pairs")

        logger.info(f"Loading {len(dev_ascii)} dev files...")
        dev_dataset = IRCDisentanglementDataset(
            ascii_files=dev_ascii,
            annotation_files=dev_ann,
            tokenizer=tokenizer,
            max_dist=args.max_dist,
            max_length=args.max_length,
        )

        logger.info(f"  Dev dataset created: {len(dev_dataset)} pairs")

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=(args.device == "cuda"),
        )

        dev_loader = DataLoader(
            dev_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(args.device == "cuda"),
        )

        return train_loader, dev_loader


def evaluate(model, dataloader, device, threshold=0.5, fp16=False):
    """Evaluate model on a dataset"""
    model.eval()

    logger.info(f"Starting evaluation on {len(dataloader.dataset)} pairs")
    start_time = datetime.now()

    all_predictions = []
    all_labels = []
    all_probs = []

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(
            tqdm(dataloader, desc="Evaluating", leave=False)
        ):
            try:
                # Move batch to device
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                features = batch["features"].to(device)
                labels = batch["labels"].to(device)

                # Handle token_type_ids if present
                token_type_ids = batch.get("token_type_ids", None)
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(device)

                # Forward pass
                with torch.cuda.amp.autocast(enabled=fp16):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        features=features,
                        labels=labels,
                    )

                # Get predictions
                probs = outputs["probs"]
                predictions = (probs >= threshold).long()

                # Store results
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

                # Accumulate loss
                if "loss" in outputs:
                    total_loss += outputs["loss"].item()
                    num_batches += 1
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.error(f"  Eval Batch {batch_idx + 1}: CUDA Out of Memory (OOM) during evaluation!")
                    if torch.cuda.is_available():
                        allocated = torch.cuda.memory_allocated(device) / (1024**2)
                        reserved = torch.cuda.memory_reserved(device) / (1024**2)
                        logger.error(f"  Memory at OOM: {allocated:.0f}/{reserved:.0f}MB")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

            # Log progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                batches_per_sec = (batch_idx + 1) / elapsed if elapsed > 0 else 0
                logger.info(
                    f"  Evaluation progress: {batch_idx + 1}/{len(dataloader)} batches ({batches_per_sec:.2f} batches/s)"
                )

    # Calculate metrics
    all_predictions = torch.tensor(all_predictions)
    all_labels = torch.tensor(all_labels)
    all_probs = torch.tensor(all_probs)

    # Calculate metrics
    tp = ((all_predictions == 1) & (all_labels == 1)).sum().item()
    fp = ((all_predictions == 1) & (all_labels == 0)).sum().item()
    tn = ((all_predictions == 0) & (all_labels == 0)).sum().item()
    fn = ((all_predictions == 0) & (all_labels == 1)).sum().item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    accuracy = (tp + tn) / len(all_labels) if len(all_labels) > 0 else 0.0

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"Evaluation complete in {elapsed:.2f}s")
    logger.info(
        f"  Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}"
    )
    logger.info(f"  TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "predictions": all_predictions,
        "labels": all_labels,
        "probs": all_probs,
    }


def train_epoch(
    model, train_loader, optimizer, scheduler, device, epoch, fp16=False, scaler=None
):
    """Train for one epoch"""
    model.train()

    logger.info(f"Starting epoch {epoch} with {len(train_loader)} batches")
    start_time = datetime.now()

    total_loss = 0.0
    num_batches = 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=True)

    for batch_idx, batch in enumerate(progress_bar):
        try:
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            features = batch["features"].to(device)
            labels = batch["labels"].to(device)

            # Handle token_type_ids if present
            token_type_ids = batch.get("token_type_ids", None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            # Forward pass
            with torch.cuda.amp.autocast(enabled=fp16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    features=features,
                    labels=labels,
                )

                loss = outputs["loss"]

            # Check for NaN/Inf loss
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(
                    f"  Batch {batch_idx + 1}: NaN or Inf loss detected! Skipping batch."
                )
                continue

            probs = outputs["probs"]
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(
                    f"  Batch {batch_idx + 1}: CUDA Out of Memory (OOM) during forward pass!"
                )
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated(device) / (1024**2)
                    reserved = torch.cuda.memory_reserved(device) / (1024**2)
                    logger.error(f"  Memory at OOM: {allocated:.0f}/{reserved:.0f}MB")

                # Clear cache and skip batch
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad
                torch.cuda.empty_cache()
                continue
            else:
                raise e

        # SMART LOGGING: Track model behavior on rare positive samples and general trends.
        # 1. Log every batch that contains a positive sample (label=1) to see how the model handles replies.
        # 2. Log every 50 batches to monitor general probability distribution and avoid log flooding.
        pos_in_batch = (labels == 1).any().item()
        if pos_in_batch or (batch_idx + 1) % 50 == 0:
            pos_labels = (labels == 1).sum().item()
            neg_labels = (labels == 0).sum().item()
            avg_prob = probs.mean().item()
            max_prob = probs.max().item()
            min_prob = probs.min().item()

            log_msg = (
                f"  Batch {batch_idx + 1} Stats: "
                f"Pos/Neg Labels: {pos_labels}/{neg_labels}, "
                f"Prob Range: [{min_prob:.4f}, {max_prob:.4f}], "
                f"Avg Prob: {avg_prob:.4f}"
            )
            if pos_in_batch:
                log_msg = "[POSITIVE BATCH] " + log_msg

            logger.info(log_msg)

        # Backward pass
        try:
            optimizer.zero_grad()
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(
                    f"  Batch {batch_idx + 1}: CUDA Out of Memory (OOM) during backward pass!"
                )
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated(device) / (1024**2)
                    reserved = torch.cuda.memory_reserved(device) / (1024**2)
                    logger.error(f"  Memory at OOM: {allocated:.0f}/{reserved:.0f}MB")

                # Clear cache and skip batch
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                continue
            else:
                raise e

        if scheduler is not None:
            scheduler.step()

        # Update metrics
        total_loss += loss.item()
        num_batches += 1

        # Update progress bar
        progress_bar.set_postfix({"loss": loss.item()})

        # Log progress every 10 batches
        if (batch_idx + 1) % 10 == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            batches_per_sec = (batch_idx + 1) / elapsed if elapsed > 0 else 0
            avg_loss_so_far = total_loss / num_batches

            # Memory logging
            mem_msg = ""
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(device) / (1024**2)
                reserved = torch.cuda.memory_reserved(device) / (1024**2)
                max_allocated = torch.cuda.max_memory_allocated(device) / (1024**2)
                mem_msg = f", GPU Mem: {allocated:.0f}/{reserved:.0f}MB (max: {max_allocated:.0f}MB)"

            logger.info(
                f"  Epoch {epoch} progress: {batch_idx + 1}/{len(train_loader)} batches "
                f"({batches_per_sec:.2f} batches/s, avg_loss={avg_loss_so_far:.4f}{mem_msg})"
            )

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"Epoch {epoch} complete in {elapsed:.2f}s, avg_loss={avg_loss:.4f}")
    return avg_loss


def save_checkpoint(model, optimizer, scheduler, epoch, args, metrics, checkpoint_dir):
    """Save training checkpoint with robust file handling"""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "args": vars(args),
        "metrics": metrics,
    }

    checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"

    # Remove existing checkpoint file if it exists (Windows compatibility)
    if checkpoint_path.exists():
        try:
            checkpoint_path.unlink()
        except OSError as e:
            logger.warning(
                f"Could not remove existing checkpoint {checkpoint_path}: {e}"
            )

    # Save to temporary file first, then rename (atomic operation)
    temp_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt.tmp"
    try:
        torch.save(checkpoint, temp_path)
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        temp_path.rename(checkpoint_path)  # Atomic on most systems
    except Exception as e:
        # Fallback: save directly if rename fails
        logger.error(f"Checkpoint save failed (temp rename): {e}")
        if temp_path.exists():
            try:
                temp_path.unlink()
            except:
                pass
        try:
            torch.save(checkpoint, checkpoint_path)
        except Exception as e2:
            logger.error(f"CRITICAL: Direct checkpoint save also failed: {e2}")

    # Also save best model
    if "f1" in metrics:
        best_path = checkpoint_dir / "best_model.pt"
        # Remove existing best model file if it exists
        if best_path.exists():
            try:
                best_path.unlink()
            except OSError as e:
                logger.warning(f"Could not remove existing best model {best_path}: {e}")
        torch.save(checkpoint, best_path)

    logger.info(f"Saved checkpoint to {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(
    checkpoint_path, model, optimizer=None, scheduler=None, device="cpu"
):
    """Load training checkpoint"""
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    metrics = checkpoint.get("metrics", {})

    logger.info(f"Loaded checkpoint from {checkpoint_path} (epoch {epoch})")
    return epoch, metrics


def log_system_info():
    """Log system diagnostics (CPU, RAM, GPU)"""
    logger.info("--- System Diagnostics ---")
    logger.info(f"OS: {platform.system()} {platform.release()}")
    logger.info(f"Python: {platform.python_version()}")

    # RAM
    virtual_mem = psutil.virtual_memory()
    logger.info(
        f"RAM: {virtual_mem.total / (1024**3):.2f} GB total, {virtual_mem.available / (1024**3):.2f} GB available"
    )

    # GPU
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"GPU {i}: {props.name}")
            logger.info(f"  Total Memory: {props.total_memory / (1024**2):.0f} MB")
            logger.info(f"  Compute Capability: {props.major}.{props.minor}")
    else:
        logger.info("GPU: No CUDA-capable device found.")
    logger.info("--------------------------")


def main():
    """Main training function"""
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save args to file
    args_path = output_dir / "args.json"
    with open(args_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    logger.info("=" * 80)
    logger.info("IRC Conversation Disentanglement Training")
    logger.info("=" * 80)
    log_system_info()
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Max dist: {args.max_dist}")
    logger.info(f"Max length: {args.max_length}")
    logger.info(f"Output dir: {args.output_dir}")
    print(f"Device: {args.device}")
    print(f"Model: {args.model_name}")
    print(f"Output dir: {output_dir}")
    print("=" * 80)

    # Set device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Load tokenizer
    logger.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    logger.info(f"Tokenizer loaded: {tokenizer.__class__.__name__}")

    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, dev_loader = create_dataloaders(args, tokenizer)

    if train_loader:
        logger.info(f"  Train dataset: {len(train_loader.dataset)} pairs")
    if dev_loader:
        logger.info(f"  Dev dataset: {len(dev_loader.dataset)} pairs")

    # Create model
    logger.info("Creating model...")
    model = create_model(
        model_name=args.model_name,
        num_features=4,
        dropout=args.dropout,
        freeze_bert=args.freeze_bert,
        device=device,
    )

    trainable, total = count_parameters(model)
    logger.info(f"  Parameters: {trainable:,} trainable, {total:,} total")
    logger.info(f"  BERT hidden size: {model.bert_hidden_size}")
    logger.info(f"  Combined size: {model.combined_size}")

    # Create optimizer and scheduler
    if args.mode in ["train", "dev-only"]:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=0.01
        )
        logger.info(f"Created AdamW optimizer with lr={args.learning_rate}")

        if train_loader:
            total_steps = len(train_loader) * args.epochs
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=total_steps,
            )
            logger.info(
                f"Created scheduler with {args.warmup_steps} warmup steps, {total_steps} total steps"
            )
        else:
            scheduler = None
    else:
        optimizer = None
        scheduler = None

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16) if args.fp16 else None

    # Resume from checkpoint if specified
    start_epoch = 1
    if args.resume_from:
        logger.info(f"Resuming from checkpoint: {args.resume_from}")
        start_epoch, _ = load_checkpoint(
            args.resume_from, model, optimizer, scheduler, device
        )
        start_epoch += 1

    # Training loop
    if args.mode in ["train", "dev-only"]:
        logger.info(f"Starting training from epoch {start_epoch}...")
        logger.info("=" * 80)

        best_f1 = 0.0
        best_epoch = 0
        no_improve_count = 0
        training_start_time = datetime.now()

        for epoch in range(start_epoch, args.epochs + 1):
            logger.info(f"Epoch {epoch}/{args.epochs}")
            logger.info("-" * 80)

            # Train
            if train_loader:
                train_loss = train_epoch(
                    model,
                    train_loader,
                    optimizer,
                    scheduler,
                    device,
                    epoch,
                    fp16=args.fp16,
                    scaler=scaler,
                )
                logger.info(f"Train Loss: {train_loss:.4f}")

            # Evaluate
            if dev_loader and epoch % args.eval_every == 0:
                logger.info("Evaluating on dev set...")
                metrics = evaluate(
                    model, dev_loader, device, args.threshold, fp16=args.fp16
                )

                logger.info(f"Dev Loss: {metrics['loss']:.4f}")
                logger.info(f"Dev Accuracy: {metrics['accuracy']:.4f}")
                logger.info(f"Dev Precision: {metrics['precision']:.4f}")
                logger.info(f"Dev Recall: {metrics['recall']:.4f}")
                logger.info(f"Dev F1: {metrics['f1']:.4f}")
                logger.info(
                    f"  TP: {metrics['tp']}, FP: {metrics['fp']}, TN: {metrics['tn']}, FN: {metrics['fn']}"
                )

                # Track best model
                if metrics["f1"] > best_f1:
                    best_f1 = metrics["f1"]
                    best_epoch = epoch
                    no_improve_count = 0
                    logger.info(f"  New best F1! Saving best model...")
                    save_checkpoint(
                        model,
                        optimizer,
                        scheduler,
                        epoch,
                        args,
                        metrics,
                        output_dir / "best",
                    )
                else:
                    no_improve_count += 1
                    logger.info(f"  No improvement for {no_improve_count} epochs")

                # Early stopping check
                if args.patience > 0 and no_improve_count >= args.patience:
                    logger.info(f"  Early stopping triggered after {epoch} epochs!")
                    # Save final checkpoint before stopping
                    metrics = {"f1": best_f1}
                    save_checkpoint(
                        model, optimizer, scheduler, epoch, args, metrics, output_dir
                    )
                    break

            # Save checkpoint
            if epoch % args.save_every == 0:
                metrics = {"f1": best_f1} if dev_loader else {}
                save_checkpoint(
                    model, optimizer, scheduler, epoch, args, metrics, output_dir
                )

        training_elapsed = (datetime.now() - training_start_time).total_seconds()
        logger.info(f"Training complete in {training_elapsed:.2f}s!")
        logger.info(f"Best F1: {best_f1:.4f} at epoch {best_epoch}")

    # Test mode
    elif args.mode == "test":
        logger.info("Running test evaluation...")
        logger.info("=" * 80)

        if dev_loader:
            metrics = evaluate(
                model, dev_loader, device, args.threshold, fp16=args.fp16
            )

            logger.info("Test Results:")
            logger.info(f"Loss: {metrics['loss']:.4f}")
            logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"Precision: {metrics['precision']:.4f}")
            logger.info(f"Recall: {metrics['recall']:.4f}")
            logger.info(f"F1: {metrics['f1']:.4f}")
            logger.info(
                f"  TP: {metrics['tp']}, FP: {metrics['fp']}, TN: {metrics['tn']}, FN: {metrics['fn']}"
            )

            # Save results
            results_path = output_dir / "test_results.json"
            with open(results_path, "w") as f:
                json.dump(
                    {
                        "loss": metrics["loss"],
                        "accuracy": metrics["accuracy"],
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "f1": metrics["f1"],
                        "tp": int(metrics["tp"]),
                        "fp": int(metrics["fp"]),
                        "tn": int(metrics["tn"]),
                        "fn": int(metrics["fn"]),
                    },
                    f,
                    indent=2,
                )

            logger.info(f"Results saved to {results_path}")

    logger.info("=" * 80)
    logger.info("Done!")
    logger.info("=" * 80)
    logger.info(f"Log file: {LOG_FILE}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        if "logger" in globals():
            if "out of memory" in str(e).lower():
                logger.error("FATAL: CUDA Out of Memory (OOM) at top level!")
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / (1024**2)
                    reserved = torch.cuda.memory_reserved() / (1024**2)
                    logger.error(f"Final Memory State: {allocated:.0f}/{reserved:.0f}MB")
             
            logger.exception("Fatal error during training:")
        else:
            import traceback
            traceback.print_exc()
        sys.exit(1)
```

---

### 4. src/evaluate.py

```python
"""
IRC Conversation Disentanglement - Checkpoint Evaluation Script

Evaluates a trained checkpoint on the dev set.
Usage: python evaluate.py --checkpoint checkpoints/best/checkpoint_epoch_3.pt

Supports threshold sweep to find optimal threshold:
Usage: python evaluate.py --checkpoint checkpoints/best/checkpoint_epoch_3.pt --sweep-thresholds
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import IRCDisentanglementDataset, load_dataset_files
from model import CrossEncoderWithFeatures, create_model
from train import evaluate

# Configure logging
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate IRC Disentanglement Checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file to evaluate",
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing train/dev/test subdirectories",
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        default="bert-base-uncased",
        help="Pretrained BERT model name",
    )
    
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Maximum token length for BERT",
    )
    
    parser.add_argument(
        "--max-dist",
        type=int,
        default=30,
        help="Maximum distance to consider for linking",
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for evaluation",
    )
    
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers",
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for binary prediction",
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for evaluation",
    )
    
    parser.add_argument(
        "--sweep-thresholds",
        action="store_true",
        help="Sweep thresholds from 0.3 to 0.9 to find optimal",
    )
    
    return parser.parse_args()


def load_checkpoint_for_eval(checkpoint_path, device):
    """Load a checkpoint for evaluation (model only, no optimizer)"""
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = create_model(
        model_name="bert-base-uncased",
        max_length=128,
        max_dist=30,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    return model


def load_dev_dataset(data_dir, max_dist, max_length, batch_size, num_workers, device):
    """Load the dev dataset for evaluation"""
    _, dev_ascii, _, dev_ann = load_dataset_files(data_dir)
    
    dev_dataset = IRCDisentanglementDataset(
        ascii_files=dev_ascii,
        annotation_files=dev_ann,
        tokenizer=None,  # Will be set later via model
        max_dist=max_dist,
        max_length=max_length,
    )
    
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )
    
    return dev_loader


def sweep_thresholds(model, dev_loader, device, thresholds=None):
    """Sweep thresholds and print metrics for each"""
    if thresholds is None:
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    logger.info("=" * 80)
    logger.info("Threshold Sweep Results")
    logger.info("=" * 80)
    
    results = []
    for thresh in thresholds:
        logger.info(f"\nEvaluating with threshold = {thresh}")
        metrics = evaluate(model, dev_loader, device, threshold=thresh)
        
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  F1: {metrics['f1']:.4f}")
        logger.info(f"  TP: {metrics['tp']}, FP: {metrics['fp']}, TN: {metrics['tn']}, FN: {metrics['fn']}")
        
        results.append({
            'threshold': thresh,
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
        })
    
    # Find best F1
    best = max(results, key=lambda x: x['f1'])
    
    logger.info("\n" + "=" * 80)
    logger.info(f"Best threshold: {best['threshold']} with F1 = {best['f1']:.4f}")
    logger.info("=" * 80)
    
    return results


def main():
    args = parse_args()
    
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Load checkpoint
    model = load_checkpoint_for_eval(args.checkpoint, device)
    
    # Load dev dataset
    logger.info("Loading dev dataset...")
    dev_loader = load_dev_dataset(
        args.data_dir,
        args.max_dist,
        args.max_length,
        args.batch_size,
        args.num_workers,
        device,
    )
    logger.info(f"Dev dataset: {len(dev_loader.dataset)} pairs")
    
    # Run evaluation
    if args.sweep_thresholds:
        sweep_thresholds(model, dev_loader, device)
    else:
        metrics = evaluate(model, dev_loader, device, threshold=args.threshold)
        
        logger.info("=" * 80)
        logger.info("Evaluation Results")
        logger.info("=" * 80)
        logger.info(f"Threshold: {args.threshold}")
        logger.info(f"Loss: {metrics['loss']:.4f}")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1: {metrics['f1']:.4f}")
        logger.info(f"  TP: {metrics['tp']}, FP: {metrics['fp']}, TN: {metrics['tn']}, FN: {metrics['fn']}")
        logger.info("=" * 80)
    
    logger.info(f"Results saved to: {LOG_FILE}")


if __name__ == "__main__":
    main()
```

---

## Project Structure

```
irc_dis_pytorch/
├── src/
│   ├── data_loader.py      # Data loading and preprocessing
│   ├── model.py            # BERT CrossEncoder with handcrafted features
│   ├── train.py            # Training script
│   └── evaluate.py         # Evaluation script
├── data/
│   ├── train/              # Training data (ASCII + annotation files)
│   ├── dev/                # Development data
│   └── test/               # Test data
├── checkpoints/            # Saved model checkpoints
├── logs/                   # Training and evaluation logs
└── research/
    └── handover.md         # This file
```

## Key Components

### 1. Data Loader ([`data_loader.py`](src/data_loader.py))
- Parses IRC log files in format: `[HH:MM] <Speaker> message`
- Creates message pairs for training/evaluation
- Computes 4 handcrafted features:
  - `time_diff_min`: Time difference in minutes (normalized)
  - `speaker_match`: 1 if same speaker, 0 otherwise
  - `pos_dist`: Position distance normalized by max_dist
  - `word_jaccard`: Jaccard similarity of word sets
- Implements lazy tokenization to save memory

### 2. Model ([`model.py`](src/model.py))
- BERT-based CrossEncoder architecture
- Extracts [CLS] token embedding (768-dim)
- Concatenates with 4 handcrafted features (772-dim)
- Linear layer + Sigmoid for binary classification
- Dynamic pos_weight for handling class imbalance

### 3. Training ([`train.py`](src/train.py))
- AdamW optimizer with learning rate scheduling
- Mixed precision training (FP16) support
- Early stopping with patience
- Checkpoint saving with atomic operations
- Comprehensive logging and memory monitoring

### 4. Evaluation ([`evaluate.py`](src/evaluate.py))
- Evaluates on dev/test sets
- Threshold sweep for optimal threshold selection
- Computes precision, recall, F1, accuracy

## Usage

### Training
```bash
python src/train.py --mode train --data-dir data --batch-size 64 --epochs 3
```

### Evaluation
```bash
python src/evaluate.py --checkpoint checkpoints/best/best_model.pt --sweep-thresholds
```

### Test Mode
```bash
python src/train.py --mode test --data-dir data
```

## Dependencies

- PyTorch
- Transformers (Hugging Face)
- NumPy
- tqdm
- psutil