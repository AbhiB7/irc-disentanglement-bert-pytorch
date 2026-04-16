"""
Unit tests for IRC Conversation Disentanglement Data Loader

Tests cover:
1. Data loading and parsing
2. Feature computation
3. Dataset creation and iteration
4. Integration with PyTorch DataLoader
"""

import os
import sys
import unittest
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
from transformers import AutoTokenizer

from data_loader import (
    IRCMessage,
    IRCConversation,
    parse_irc_line,
    extract_targets,
    load_conversation,
    compute_features,
    IRCDisentanglementDataset,
    load_dataset_files,
)


class TestIRCMessageParsing(unittest.TestCase):
    """Test IRC message parsing functions"""

    def test_parse_regular_message(self):
        """Test parsing a regular IRC message"""
        line = "[12:34] <user1> Hello world"
        timestamp, speaker, text, is_system = parse_irc_line(line)
        
        self.assertEqual(timestamp, (12, 34))
        self.assertEqual(speaker, "user1")
        self.assertEqual(text, "Hello world")
        self.assertFalse(is_system)

    def test_parse_system_message(self):
        """Test parsing a system message"""
        line = "=== System message ==="
        timestamp, speaker, text, is_system = parse_irc_line(line)
        
        self.assertIsNone(timestamp)
        self.assertEqual(speaker, "SYSTEM")
        self.assertEqual(text, "=== System message ===")
        self.assertTrue(is_system)

    def test_parse_unknown_format(self):
        """Test parsing an unknown format message"""
        line = "Some random text"
        timestamp, speaker, text, is_system = parse_irc_line(line)
        
        self.assertIsNone(timestamp)
        self.assertEqual(speaker, "UNKNOWN")
        self.assertEqual(text, "Some random text")
        self.assertTrue(is_system)


class TestTargetExtraction(unittest.TestCase):
    """Test target user extraction from messages"""

    def test_extract_targets_simple(self):
        """Test extracting simple mentions"""
        text = "Hello @user1 and @user2"
        users = {"user1", "user2", "user3"}
        targets = extract_targets(text, users)
        
        self.assertEqual(targets, {"user1", "user2"})

    def test_extract_targets_case_insensitive(self):
        """Test case-insensitive matching"""
        text = "Hello USER1 and User2"
        users = {"user1", "user2", "user3"}
        targets = extract_targets(text, users)
        
        self.assertEqual(targets, {"user1", "user2"})

    def test_extract_targets_no_mentions(self):
        """Test when no users are mentioned"""
        text = "Hello everyone"
        users = {"user1", "user2"}
        targets = extract_targets(text, users)
        
        self.assertEqual(targets, set())


class TestFeatureComputation(unittest.TestCase):
    """Test feature computation for message pairs"""

    def setUp(self):
        """Set up test data"""
        # Create test messages
        self.msg1 = IRCMessage(
            index=0,
            timestamp=(10, 30),
            speaker="user1",
            text="Hello world",
            is_system=False,
            is_bot=False,
            targets=set(),
            last_from_same_user=None,
            next_from_same_user=None
        )
        
        self.msg2 = IRCMessage(
            index=1,
            timestamp=(10, 31),
            speaker="user1",
            text="How are you?",
            is_system=False,
            is_bot=False,
            targets=set(),
            last_from_same_user=0,
            next_from_same_user=None
        )
        
        self.msg3 = IRCMessage(
            index=2,
            timestamp=(10, 35),
            speaker="user2",
            text="I'm fine thanks",
            is_system=False,
            is_bot=False,
            targets=set(),
            last_from_same_user=None,
            next_from_same_user=None
        )
        
        # Create test conversation
        self.conv = IRCConversation(
            name="test",
            messages=[self.msg1, self.msg2, self.msg3],
            gold_links={},
            user_message_indices={"user1": [0, 1], "user2": [2]}
        )

    def test_compute_features_same_speaker(self):
        """Test features when messages are from same speaker"""
        features = compute_features(self.msg1, self.msg2, self.conv)
        
        # Should have 4 features
        self.assertEqual(len(features), 4)
        
        # Speaker match should be 1.0
        self.assertEqual(features[1], 1.0)
        
        # Position distance should be 1
        self.assertAlmostEqual(features[2], 1.0 / 101.0, places=5)

    def test_compute_features_different_speaker(self):
        """Test features when messages are from different speakers"""
        features = compute_features(self.msg1, self.msg3, self.conv)
        
        # Speaker match should be 0.0
        self.assertEqual(features[1], 0.0)
        
        # Position distance should be 2
        self.assertAlmostEqual(features[2], 2.0 / 101.0, places=5)

    def test_compute_features_time_difference(self):
        """Test time difference feature"""
        features = compute_features(self.msg1, self.msg3, self.conv)
        
        # Time difference: 5 minutes (10:35 - 10:30)
        # Normalized: min(5/60, 1.0) = 0.0833...
        expected_time_diff = min(5.0 / 60.0, 1.0)
        self.assertAlmostEqual(features[0], expected_time_diff, places=5)

    def test_compute_features_word_jaccard(self):
        """Test word Jaccard similarity"""
        msg_a = IRCMessage(
            index=0,
            timestamp=(10, 30),
            speaker="user1",
            text="hello world test",
            is_system=False,
            is_bot=False,
            targets=set(),
            last_from_same_user=None,
            next_from_same_user=None
        )
        
        msg_b = IRCMessage(
            index=1,
            timestamp=(10, 31),
            speaker="user2",
            text="hello test message",
            is_system=False,
            is_bot=False,
            targets=set(),
            last_from_same_user=None,
            next_from_same_user=None
        )
        
        conv = IRCConversation(
            name="test",
            messages=[msg_a, msg_b],
            gold_links={},
            user_message_indices={"user1": [0], "user2": [1]}
        )
        
        features = compute_features(msg_a, msg_b, conv)
        
        # Words: msg_a = {hello, world, test}, msg_b = {hello, test, message}
        # Intersection: {hello, test} = 2
        # Union: {hello, world, test, message} = 4
        # Jaccard: 2/4 = 0.5
        self.assertAlmostEqual(features[3], 0.5, places=5)


class TestLoadConversation(unittest.TestCase):
    """Test conversation loading from files"""

    def test_load_sample_conversation(self):
        """Test loading a real conversation file"""
        # Use the sample file from the data directory
        sample_ascii = "data/dev/2004-11-15_03.ascii.txt"
        sample_ann = "data/dev/2004-11-15_03.annotation.txt"
        
        if os.path.exists(sample_ascii) and os.path.exists(sample_ann):
            conv = load_conversation(sample_ascii, sample_ann)
            
            self.assertEqual(conv.name, "2004-11-15_03")
            self.assertGreater(len(conv.messages), 0)
            self.assertGreater(len(conv.gold_links), 0)
            
            # Check that messages have correct structure
            for msg in conv.messages:
                self.assertIsInstance(msg.index, int)
                self.assertIsInstance(msg.speaker, str)
                self.assertIsInstance(msg.text, str)
                self.assertIsInstance(msg.is_system, bool)
                self.assertIsInstance(msg.is_bot, bool)
        else:
            self.skipTest("Sample conversation files not found")


class TestIRCDataset(unittest.TestCase):
    """Test IRCDisentanglementDataset"""

    def setUp(self):
        """Set up test dataset"""
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        # Use a single dev file for testing
        self.ascii_files = ["data/dev/2004-11-15_03.ascii.txt"]
        self.annotation_files = ["data/dev/2004-11-15_03.annotation.txt"]

    def test_dataset_creation(self):
        """Test creating the dataset"""
        if not os.path.exists(self.ascii_files[0]):
            self.skipTest("Test data files not found")
            return
        
        dataset = IRCDisentanglementDataset(
            ascii_files=self.ascii_files,
            annotation_files=self.annotation_files,
            tokenizer=self.tokenizer,
            max_dist=101,
            max_length=128
        )
        
        # Check dataset properties
        self.assertGreater(len(dataset), 0)
        self.assertGreater(len(dataset.pairs), 0)
        self.assertGreater(len(dataset.conversations), 0)

    def test_dataset_getitem(self):
        """Test getting items from dataset"""
        if not os.path.exists(self.ascii_files[0]):
            self.skipTest("Test data files not found")
            return
        
        dataset = IRCDisentanglementDataset(
            ascii_files=self.ascii_files,
            annotation_files=self.annotation_files,
            tokenizer=self.tokenizer,
            max_dist=101,
            max_length=128
        )
        
        # Get first item
        item = dataset[0]
        
        # Check item structure
        self.assertIn('input_ids', item)
        self.assertIn('attention_mask', item)
        self.assertIn('features', item)
        self.assertIn('labels', item)
        
        # Check tensor shapes
        self.assertEqual(item['input_ids'].dim(), 1)
        self.assertEqual(item['attention_mask'].dim(), 1)
        self.assertEqual(item['features'].dim(), 1)
        self.assertEqual(item['features'].shape[0], 4)  # 4 features

    def test_dataset_with_dataloader(self):
        """Test using dataset with PyTorch DataLoader"""
        if not os.path.exists(self.ascii_files[0]):
            self.skipTest("Test data files not found")
            return
        
        dataset = IRCDisentanglementDataset(
            ascii_files=self.ascii_files,
            annotation_files=self.annotation_files,
            tokenizer=self.tokenizer,
            max_dist=101,
            max_length=128
        )
        
        # Create DataLoader
        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
        
        # Get one batch
        batch = next(iter(dataloader))
        
        # Check batch structure
        self.assertIn('input_ids', batch)
        self.assertIn('attention_mask', batch)
        self.assertIn('features', batch)
        self.assertIn('labels', batch)
        
        # Check batch shapes
        self.assertEqual(batch['input_ids'].shape[0], 2)  # batch_size
        self.assertEqual(batch['features'].shape[1], 4)   # 4 features


class TestLoadDatasetFiles(unittest.TestCase):
    """Test loading dataset file paths"""

    def test_load_dev_files(self):
        """Test loading dev split file paths"""
        if not os.path.exists("data/dev"):
            self.skipTest("Dev data directory not found")
            return
        
        ascii_files, annotation_files = load_dataset_files("data", split="dev")
        
        self.assertGreater(len(ascii_files), 0)
        self.assertEqual(len(ascii_files), len(annotation_files))
        
        # Check file extensions
        for file in ascii_files:
            self.assertTrue(file.endswith('.ascii.txt'))
        
        for file in annotation_files:
            self.assertTrue(file.endswith('.annotation.txt'))


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
