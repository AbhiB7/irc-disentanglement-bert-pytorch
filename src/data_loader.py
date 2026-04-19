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
        is_test: bool = False,
        test_start: int = 1000,
        test_end: int = 1000000,
    ):
        """
        Args:
            ascii_files: List of ASCII file paths
            annotation_files: List of annotation file paths (parallel to ascii_files)
            tokenizer: BERT tokenizer
            max_dist: Maximum distance to consider for linking (default 30)
            max_length: Maximum token length for BERT
            is_test: If True, generate pairs for all messages (no gold labels)
            test_start/end: For test mode, which messages to process
        """
        assert len(ascii_files) == len(annotation_files), "File lists must match"

        self.tokenizer = tokenizer
        self.max_dist = max_dist
        self.max_length = max_length
        self.is_test = is_test
        self.test_start = test_start
        self.test_end = test_end

        # Load all conversations
        self.conversations = []
        self.conversation_map = (
            []
        )  # Maps pair index to (conv_idx, msg_i_idx, msg_j_idx)
        self.pairs = []  # List of (text_pair, label, features)

        logger.info(
            f"Initializing IRCDisentanglementDataset with {len(ascii_files)} files"
        )
        logger.info(
            f"  max_dist={max_dist}, max_length={max_length}, is_test={is_test}"
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

            # Early exit if we've reached test_end pairs (for test mode)
            if self.is_test and len(self.pairs) >= self.test_end:
                logger.info(
                    f"  Reached test_end limit ({self.test_end} pairs), stopping early"
                )
                break

        # Truncate pairs to test_end if specified (for test mode)
        if self.is_test and self.test_end < len(self.pairs):
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
        if self.is_test:
            start_idx = self.test_start
            end_idx = min(self.test_end, len(messages))
            process_indices = range(start_idx, end_idx)
            logger.info(
                f"  Creating pairs for {conv.name}: messages {start_idx} to {end_idx} (test mode)"
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

                # Create pair
                text_pair = [msg_j.text, msg_i.text]  # [parent, child]

                # Label: 1 if j is a gold parent of i, 0 otherwise
                label = 1.0 if (i in gold_links and j in gold_links[i]) else 0.0

                # For test mode, we don't have gold labels
                if self.is_test:
                    label = -1.0  # Placeholder

                # Compute features
                features = compute_features(
                    msg_j, msg_i, conv, max_dist=self.max_dist
                )  # parent, child

                # Store
                self.pairs.append((text_pair, label, features))
                self.conversation_map.append((conv_idx, i, j))

        pairs_added = len(self.pairs) - pairs_before
        logger.info(f"  Created {pairs_added} pairs for {conv.name}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        text_pair, label, features = self.pairs[idx]

        # Tokenize the pair for BERT CrossEncoder
        encoding = self.tokenizer(
            text_pair[0],
            text_pair[1],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Convert to dict and add features
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding.get("token_type_ids", None),
            "features": torch.tensor(features, dtype=torch.float32),
            "labels": (
                torch.tensor(label, dtype=torch.float32)
                if label != -1
                else torch.tensor(0.0, dtype=torch.float32)
            ),
        }

        # Remove token_type_ids if None
        if item["token_type_ids"] is None:
            del item["token_type_ids"]
        else:
            item["token_type_ids"] = item["token_type_ids"].squeeze(0)

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
