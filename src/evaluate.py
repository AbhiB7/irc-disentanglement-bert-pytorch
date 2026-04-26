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
