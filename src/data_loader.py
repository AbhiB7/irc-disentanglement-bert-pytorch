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
    """Represents a single IRC message with metadata
    Tested: tests/test_data_loader.py (parse_irc_line → builds IRCMessage)"""

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
    """Represents a single IRC log file with messages and gold links
    Tested: tests/test_create_samples.py & tests/test_data_loader.py (built & used by both)
    """

    name: str
    messages: List[IRCMessage]
    gold_links: Dict[int, List[int]]  # child -> list of parent indices
    # Additional metadata for feature extraction
    user_message_indices: Dict[str, List[int]]  # user -> list of message indices


def parse_irc_line(line: str) -> Tuple[Optional[Tuple[int, int]], str, str, bool]:
    """
    Parse an IRC line in format: [HH:MM] <Speaker> message
    Returns: (timestamp, speaker, text, is_system)
    Tested: tests/test_data_loader.py (Test 8: 4 cases — normal, system, underscore name, garbage)
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
    Indirectly verified by tests/test_create_samples.py Test 4 (directedness)
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
    Tested: tests/test_load_conversation.py (4 tests — basic, system msgs, empty ann, self-link)
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
    max_dist: int = 50,
) -> List[float]:
    """Tested: tests/test_create_samples.py
    Compute 5 handcrafted features as per project plan:
    1. time_diff_min: Time difference in minutes (capped at -100)
    2. speaker_match: 1 if same speaker, 0 otherwise
    3. pos_dist: Position distance (j - i) normalized by max_dist
    4. word_jaccard: Jaccard similarity of word sets
    5. directedness: 1 if child message mentions parent's speaker, 0 otherwise

    Returns: List of 5 feature values
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

    # 5. Directedness: child mentions parent's speaker
    directedness = 1.0 if msg_i.speaker in msg_j.targets else 0.0

    return [time_diff_norm, speaker_match, pos_dist_norm, jaccard, directedness]


class IRCDisentanglementDataset(Dataset):
    """
    PyTorch Dataset for IRC conversation disentanglement.
    Creates multiclass samples (which candidate is the parent?) - one sample per candidate.
    """

    def __init__(
        self,
        ascii_files: List[str],
        annotation_files: List[str],
        tokenizer,
        max_dist: int = 50,
        max_length: int = 128,
        skip_labels: bool = False,
        test_start: int = 0,
        test_end: int = 1000000000,  # Default to 1 Billion (effectively no limit).
        # Note: Previous limit of 1M was too low for full dataset.
    ):
        """
        Args:
            ascii_files: List of ASCII file paths
            annotation_files: List of annotation file paths (parallel to ascii_files)
            tokenizer: BERT tokenizer
            max_dist: Maximum distance to consider for linking (default 50)
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
        )  # Maps sample index to (conv_idx, msg_i_idx, candidate_idx)
        self.samples = (
            []
        )  # List of (parent_text, child_text, features, label) - raw text for lazy tokenization

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
                disable=True,
            )
        ):
            conv = load_conversation(ascii_path, ann_path)
            self.conversations.append(conv)

            # Create multiclass samples (one per candidate)
            self._create_samples_for_conversation(conv, len(self.conversations) - 1)

            logger.info(
                f"  File {idx+1}/{len(ascii_files)}: {conv.name} - {len(conv.messages)} messages, {len(self.samples)} total samples so far"
            )

            # Early exit if we've reached test_end samples (if limiting)
            if self.test_end < 1000000000 and len(self.samples) >= self.test_end:
                logger.info(
                    f"  Reached test_end limit ({self.test_end} samples), stopping early"
                )
                break

        # Truncate samples to test_end if specified
        if self.test_end < 1000000000 and self.test_end < len(self.samples):
            logger.info(
                f"Truncating samples from {len(self.samples)} to {self.test_end} (test_end limit)"
            )
            self.samples = self.samples[: self.test_end]
            self.conversation_map = self.conversation_map[: self.test_end]

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(
            f"Dataset initialization complete: {len(self.samples)} samples from {len(self.conversations)} conversations in {elapsed:.2f}s"
        )

    def _create_samples_for_conversation(self, conv: IRCConversation, conv_idx: int):
        """Tested: tests/test_create_samples.py
        Create multiclass samples for a conversation (one sample per child message)"""
        messages = conv.messages
        gold_links = conv.gold_links

        # Determine which messages to process
        if self.test_end < 1000000000:
            start_idx = self.test_start
            end_idx = min(self.test_end, len(messages))
            process_indices = range(start_idx, end_idx)
            logger.info(
                f"  Creating samples for {conv.name}: messages {start_idx} to {end_idx} (limited)"
            )
        else:
            process_indices = range(len(messages))
            logger.info(
                f"  Creating samples for {conv.name}: all {len(messages)} messages"
            )

        samples_before = len(self.samples)

        # Use tqdm for progress bar on message iteration
        for i in tqdm(
            process_indices,
            desc=f"  Samples for {conv.name}",
            leave=False,
            disable=True,
        ):
            msg_i = messages[i]

            # Collect candidates within max_dist
            candidates = []
            candidate_indices = []  # (conv_idx, msg_i_idx, candidate_idx)

            for j in range(max(0, i - self.max_dist + 1), i + 1):
                msg_j = messages[j]

                # Skip system messages as parents (except self-links)
                if j != i and msg_j.is_system:
                    continue

                candidates.append(msg_j.text)
                candidate_indices.append((conv_idx, i, j))

            if not candidates:
                continue  # Skip messages with no valid candidates

            # Find gold parent index
            gold_parent_idx = -1
            if i in gold_links:
                for candidate_idx, (conv_idx_c, i_c, j_c) in enumerate(
                    candidate_indices
                ):
                    if j_c in gold_links[i]:
                        gold_parent_idx = candidate_idx
                        break

            # For blind test mode, we don't have gold labels
            if self.skip_labels:
                gold_parent_idx = -1  # Placeholder

            # Skip samples where gold parent is outside the search window
            # (no valid answer to learn from)
            if not self.skip_labels and gold_parent_idx < 0:
                continue

            # Compute features for each candidate and store per-candidate features
            per_candidate_features = []
            for candidate_idx, (conv_idx_c, i_c, j_c) in enumerate(candidate_indices):
                msg_j = messages[j_c]
                features = compute_features(
                    msg_j, msg_i, conv, max_dist=self.max_dist
                )  # parent, child
                per_candidate_features.append(features)

            # Store ONE sample per child message with all candidate features
            # Store parent_text as the gold parent text (or first candidate if no gold)
            if gold_parent_idx >= 0 and gold_parent_idx < len(candidates):
                parent_text = candidates[gold_parent_idx]
            else:
                parent_text = candidates[0] if candidates else ""

            # Store all candidate features as a 2D tensor [C, num_features]
            all_features = torch.tensor(per_candidate_features)  # [C, 5]

            # Store sample - tokenization happens in __getitem__ (lazy)
            self.samples.append(
                (parent_text, msg_i.text, all_features, gold_parent_idx)
            )
            self.conversation_map.append((conv_idx, i, candidate_indices))

        samples_added = len(self.samples) - samples_before
        logger.info(f"  Created {samples_added} samples for {conv.name}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Tested: tests/test_data_loader.py (Tests 1-7: structure, labels, counts, features, determinism)"""
        # Get sample data: (parent_text, child_text, features, gold_parent_idx)
        parent_text, child_text, features, gold_parent_idx = self.samples[idx]

        # Get conversation and message indices from conversation_map
        conv_idx, msg_i_idx, candidate_indices = self.conversation_map[idx]
        conv = self.conversations[conv_idx]
        msg_i = conv.messages[msg_i_idx]  # child message

        # candidate_indices is a list of (conv_idx, i, j) tuples
        # Extract just the j indices (candidate message indices)
        all_candidate_indices = [j for (_, _, j) in candidate_indices]

        # Get candidate texts
        candidate_texts = [conv.messages[j].text for j in all_candidate_indices]

        # Tokenize all candidate messages
        # Each candidate is a separate sequence
        candidate_encodings = []
        for c_text in candidate_texts:
            encoding = self.tokenizer(
                c_text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            candidate_encodings.append(encoding)

        # Prepare batch input_ids and attention_mask
        # The model expects input_ids of shape [C, seq_len] where C is number of candidates
        input_ids_list = []
        attention_mask_list = []

        # Add all candidate sequences
        for cand_enc in candidate_encodings:
            input_ids_list.append(cand_enc["input_ids"].squeeze(0))  # Remove batch dim
            attention_mask_list.append(cand_enc["attention_mask"].squeeze(0))

        # Stack all candidate inputs: [C, seq_len]
        all_candidate_input_ids = torch.stack(input_ids_list)
        all_candidate_attention_mask = torch.stack(attention_mask_list)

        # Token type IDs might not be necessary for DeBERTa, but include if available
        if "token_type_ids" in candidate_encodings[0]:
            all_candidate_token_type_ids = torch.stack(
                [enc["token_type_ids"].squeeze(0) for enc in candidate_encodings]
            )
        else:
            all_candidate_token_type_ids = None

        # features is already a 2D tensor [C, 5] from _create_samples_for_conversation
        # gold_parent_idx is the index of the correct candidate in the candidate_indices list

        item = {
            "input_ids": all_candidate_input_ids,  # [C, seq_len]
            "attention_mask": all_candidate_attention_mask,  # [C, seq_len]
            "features": features,  # Already a torch tensor [C, 5]
            "labels": torch.tensor(gold_parent_idx, dtype=torch.long),
        }

        if all_candidate_token_type_ids is not None:
            item["token_type_ids"] = all_candidate_token_type_ids

        return item


def load_dataset_files(
    data_dir: str, split: str = "train"
) -> Tuple[List[str], List[str]]:
    """
    Load file paths for a given split.
    Returns: (ascii_files, annotation_files)
    Tested: tests/test_load_conversation.py (Test 5: train, dev, invalid split)
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
