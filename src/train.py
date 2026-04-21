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
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
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
        help="Maximum distance to consider for linking",
    )

    # Training hyperparameters
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size for training"
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
        "--threshold", type=float, default=0.3, help="Threshold for binary prediction"
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

        # Determine if we should use test mode (is_test=True) based on test_start/test_end
        # If test_end is less than default (1000000), user wants limited pairs
        is_test_mode = args.test_end < 1000000 or args.test_start > 0

        dev_dataset = IRCDisentanglementDataset(
            ascii_files=dev_ascii,
            annotation_files=dev_ann,
            tokenizer=tokenizer,
            max_dist=args.max_dist,
            max_length=args.max_length,
            is_test=is_test_mode,
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
        train_dataset = IRCDisentanglementDataset(
            ascii_files=train_ascii,
            annotation_files=train_ann,
            tokenizer=tokenizer,
            max_dist=args.max_dist,
            max_length=args.max_length,
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
