"""
Tests for train.py parse_args() function.

Verifies that command-line argument parsing works correctly:
- Default values are as expected
- Custom values are parsed correctly
- Type conversions work (int, float, bool)
- Choices are enforced

Run with: python -m pytest tests/test_parse_args.py -v
"""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pytest
from train import parse_args


def _parse_args(argv_list):
    """Helper to call parse_args with a mock sys.argv"""
    with patch('sys.argv', ['train.py'] + argv_list):
        return parse_args()


class TestParseArgsDefaults:
    """Test that parse_args() returns correct default values"""

    def test_default_mode(self):
        """Test default mode is 'train'"""
        args = _parse_args([])
        assert args.mode == "train", f"Expected mode='train', got '{args.mode}'"

    def test_default_data_dir(self):
        """Test default data-dir is 'data'"""
        args = _parse_args([])
        assert args.data_dir == "data", f"Expected data_dir='data', got '{args.data_dir}'"

    def test_default_model_name(self):
        """Test default model-name is 'microsoft/deberta-v3-base'"""
        args = _parse_args([])
        assert args.model_name == "microsoft/deberta-v3-base", \
            f"Expected model_name='microsoft/deberta-v3-base', got '{args.model_name}'"

    def test_default_max_length(self):
        """Test default max-length is 128"""
        args = _parse_args([])
        assert args.max_length == 128, f"Expected max_length=128, got {args.max_length}"

    def test_default_max_dist(self):
        """Test default max-dist is 50"""
        args = _parse_args([])
        assert args.max_dist == 50, f"Expected max_dist=50, got {args.max_dist}"

    def test_default_batch_size(self):
        """Test default batch-size is 64"""
        args = _parse_args([])
        assert args.batch_size == 64, f"Expected batch_size=64, got {args.batch_size}"

    def test_default_learning_rate(self):
        """Test default learning-rate is 5e-5"""
        args = _parse_args([])
        assert args.learning_rate == 5e-5, f"Expected learning_rate=5e-5, got {args.learning_rate}"

    def test_default_epochs(self):
        """Test default epochs is 3"""
        args = _parse_args([])
        assert args.epochs == 3, f"Expected epochs=3, got {args.epochs}"

    def test_default_warmup_ratio(self):
        """Test default warmup-ratio is 0.1"""
        args = _parse_args([])
        assert args.warmup_ratio == 0.1, f"Expected warmup_ratio=0.1, got {args.warmup_ratio}"

    def test_default_dropout(self):
        """Test default dropout is 0.1"""
        args = _parse_args([])
        assert args.dropout == 0.1, f"Expected dropout=0.1, got {args.dropout}"

    def test_default_freeze_bert(self):
        """Test default freeze-bert is False"""
        args = _parse_args([])
        assert args.freeze_bert is False, f"Expected freeze_bert=False, got {args.freeze_bert}"

    def test_default_patience(self):
        """Test default patience is 3"""
        args = _parse_args([])
        assert args.patience == 3, f"Expected patience=3, got {args.patience}"

    def test_default_output_dir(self):
        """Test default output-dir is 'checkpoints'"""
        args = _parse_args([])
        assert args.output_dir == "checkpoints", \
            f"Expected output_dir='checkpoints', got '{args.output_dir}'"

    def test_default_save_every(self):
        """Test default save-every is 1"""
        args = _parse_args([])
        assert args.save_every == 1, f"Expected save_every=1, got {args.save_every}"

    def test_default_eval_every(self):
        """Test default eval-every is 1"""
        args = _parse_args([])
        assert args.eval_every == 1, f"Expected eval_every=1, got {args.eval_every}"

    def test_default_num_workers(self):
        """Test default num-workers is 0"""
        args = _parse_args([])
        assert args.num_workers == 0, f"Expected num_workers=0, got {args.num_workers}"

    def test_default_fp16(self):
        """Test default fp16 is False"""
        args = _parse_args([])
        assert args.fp16 is False, f"Expected fp16=False, got {args.fp16}"

    def test_default_test_start(self):
        """Test default test-start is 0"""
        args = _parse_args([])
        assert args.test_start == 0, f"Expected test_start=0, got {args.test_start}"

    def test_default_test_end(self):
        """Test default test-end is 1000000"""
        args = _parse_args([])
        assert args.test_end == 1000000, f"Expected test_end=1000000, got {args.test_end}"

    def test_default_resume_from(self):
        """Test default resume-from is None"""
        args = _parse_args([])
        assert args.resume_from is None, f"Expected resume_from=None, got {args.resume_from}"


class TestParseArgsCustom:
    """Test that parse_args() correctly parses custom values"""

    def test_custom_mode(self):
        """Test custom mode is parsed correctly"""
        args = _parse_args(["--mode", "test"])
        assert args.mode == "test", f"Expected mode='test', got '{args.mode}'"

    def test_custom_data_dir(self):
        """Test custom data-dir is parsed correctly"""
        args = _parse_args(["--data-dir", "/custom/path"])
        assert args.data_dir == "/custom/path", \
            f"Expected data_dir='/custom/path', got '{args.data_dir}'"

    def test_custom_model_name(self):
        """Test custom model-name is parsed correctly"""
        args = _parse_args(["--model-name", "prajjwal1/bert-tiny"])
        assert args.model_name == "prajjwal1/bert-tiny", \
            f"Expected model_name='prajjwal1/bert-tiny', got '{args.model_name}'"

    def test_custom_batch_size(self):
        """Test custom batch-size is parsed as int"""
        args = _parse_args(["--batch-size", "16"])
        assert args.batch_size == 16, f"Expected batch_size=16, got {args.batch_size}"
        assert isinstance(args.batch_size, int), "batch_size should be int"

    def test_custom_learning_rate(self):
        """Test custom learning-rate is parsed as float"""
        args = _parse_args(["--learning-rate", "2e-5"])
        assert args.learning_rate == 2e-5, f"Expected learning_rate=2e-5, got {args.learning_rate}"
        assert isinstance(args.learning_rate, float), "learning_rate should be float"

    def test_custom_epochs(self):
        """Test custom epochs is parsed as int"""
        args = _parse_args(["--epochs", "10"])
        assert args.epochs == 10, f"Expected epochs=10, got {args.epochs}"

    def test_custom_max_length(self):
        """Test custom max-length is parsed as int"""
        args = _parse_args(["--max-length", "256"])
        assert args.max_length == 256, f"Expected max_length=256, got {args.max_length}"

    def test_custom_max_dist(self):
        """Test custom max-dist is parsed as int"""
        args = _parse_args(["--max-dist", "100"])
        assert args.max_dist == 100, f"Expected max_dist=100, got {args.max_dist}"

    def test_custom_dropout(self):
        """Test custom dropout is parsed as float"""
        args = _parse_args(["--dropout", "0.3"])
        assert args.dropout == 0.3, f"Expected dropout=0.3, got {args.dropout}"

    def test_custom_patience(self):
        """Test custom patience is parsed as int"""
        args = _parse_args(["--patience", "5"])
        assert args.patience == 5, f"Expected patience=5, got {args.patience}"

    def test_custom_output_dir(self):
        """Test custom output-dir is parsed correctly"""
        args = _parse_args(["--output-dir", "my_checkpoints"])
        assert args.output_dir == "my_checkpoints", \
            f"Expected output_dir='my_checkpoints', got '{args.output_dir}'"

    def test_custom_num_workers(self):
        """Test custom num-workers is parsed as int"""
        args = _parse_args(["--num-workers", "4"])
        assert args.num_workers == 4, f"Expected num_workers=4, got {args.num_workers}"

    def test_custom_test_start(self):
        """Test custom test-start is parsed as int"""
        args = _parse_args(["--test-start", "100"])
        assert args.test_start == 100, f"Expected test_start=100, got {args.test_start}"

    def test_custom_test_end(self):
        """Test custom test-end is parsed as int"""
        args = _parse_args(["--test-end", "500"])
        assert args.test_end == 500, f"Expected test_end=500, got {args.test_end}"

    def test_custom_resume_from(self):
        """Test custom resume-from is parsed correctly"""
        args = _parse_args(["--resume-from", "checkpoints/checkpoint_epoch_2.pt"])
        assert args.resume_from == "checkpoints/checkpoint_epoch_2.pt", \
            f"Expected resume_from='checkpoints/checkpoint_epoch_2.pt', got '{args.resume_from}'"


class TestParseArgsFlags:
    """Test that boolean flags are parsed correctly"""

    def test_freeze_bert_flag(self):
        """Test --freeze-bert flag sets freeze_bert=True"""
        args = _parse_args(["--freeze-bert"])
        assert args.freeze_bert is True, f"Expected freeze_bert=True, got {args.freeze_bert}"

    def test_fp16_flag(self):
        """Test --fp16 flag sets fp16=True"""
        args = _parse_args(["--fp16"])
        assert args.fp16 is True, f"Expected fp16=True, got {args.fp16}"


class TestParseArgsChoices:
    """Test that argument choices are enforced"""

    def test_invalid_mode_raises_error(self):
        """Test that an invalid mode raises SystemExit"""
        with pytest.raises(SystemExit):
            _parse_args(["--mode", "invalid_mode"])

    def test_valid_modes(self):
        """Test that all valid modes are accepted"""
        for mode in ["train", "dev-only", "test"]:
            args = _parse_args(["--mode", mode])
            assert args.mode == mode, f"Expected mode='{mode}', got '{args.mode}'"


class TestParseArgsDevice:
    """Test device argument parsing"""

    def test_custom_device_cpu(self):
        """Test --device cpu is parsed correctly"""
        args = _parse_args(["--device", "cpu"])
        assert args.device == "cpu", f"Expected device='cpu', got '{args.device}'"

    def test_custom_device_cuda(self):
        """Test --device cuda is parsed correctly"""
        args = _parse_args(["--device", "cuda"])
        assert args.device == "cuda", f"Expected device='cuda', got '{args.device}'"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])