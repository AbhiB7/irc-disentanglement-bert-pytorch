"""
IRC Conversation Disentanglement - Checkpoint Evaluation Script

Evaluates a trained checkpoint on the dev set (multiclass mode).
Usage: python evaluate.py --checkpoint checkpoints/best/checkpoint_epoch_3.pt

Note: Multiclass mode uses argmax for predictions (no threshold needed).
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
from train import evaluate, collate_fn

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
        description="Evaluate IRC Disentanglement Checkpoint (Multiclass Mode)",
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
        "--split",
        type=str,
        default="dev",
        choices=["dev", "test"],
        help="Which split to evaluate on (dev or test)",
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        default="microsoft/deberta-v3-base",
        help="Pretrained model name (default: DeBERTa-v3-base)",
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
        default=50,
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
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for evaluation",
    )
    
    return parser.parse_args()


def load_checkpoint_for_eval(checkpoint_path, device):
    """Load a checkpoint for evaluation (model only, no optimizer)"""
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Read model name from checkpoint args, fall back to DeBERTa-v3-base
    ckpt_args = checkpoint.get("args", {})
    model_name = ckpt_args.get("model_name", "microsoft/deberta-v3-base")
    num_features = ckpt_args.get("num_features", 5)
    dropout = ckpt_args.get("dropout", 0.1)
    
    model = create_model(
        model_name=model_name,
        num_features=num_features,
        dropout=dropout,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')} (model={model_name})")
    return model


def load_dev_dataset(data_dir, split, max_dist, max_length, batch_size, num_workers, device, tokenizer):
    """Load the dataset for evaluation (dev or test split)"""
    dev_ascii, dev_ann = load_dataset_files(data_dir, split=split)
    
    dev_dataset = IRCDisentanglementDataset(
        ascii_files=dev_ascii,
        annotation_files=dev_ann,
        tokenizer=tokenizer,
        max_dist=max_dist,
        max_length=max_length,
    )
    
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        collate_fn=collate_fn,
    )
    
    return dev_loader


def main():
    args = parse_args()
    
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    logger.info(f"Loaded tokenizer: {tokenizer.__class__.__name__}")
    
    # Load checkpoint
    model = load_checkpoint_for_eval(args.checkpoint, device)
    
    # Load dataset
    logger.info(f"Loading {args.split} dataset...")
    dev_loader = load_dev_dataset(
        args.data_dir,
        args.split,
        args.max_dist,
        args.max_length,
        args.batch_size,
        args.num_workers,
        device,
        tokenizer,
    )
    logger.info(f"{args.split} dataset: {len(dev_loader.dataset)} samples")
    
    # Run evaluation (multiclass mode - no threshold needed)
    metrics = evaluate(model, dev_loader, device)
    
    logger.info("=" * 80)
    logger.info("Evaluation Results (Multiclass Mode)")
    logger.info("=" * 80)
    logger.info(f"Loss: {metrics['loss']:.4f}")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"F1: {metrics['f1']:.4f}")
    logger.info("=" * 80)
    
    logger.info(f"Results saved to: {LOG_FILE}")


if __name__ == "__main__":
    main()
