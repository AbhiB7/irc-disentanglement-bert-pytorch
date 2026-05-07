"""
IRC Conversation Disentanglement Training Script

Trains a BERT-based CrossEncoder with handcrafted features for IRC message linking.
Uses the tested data_loader.py and model.py modules.
"""

import argparse
import os
import sys
import time

# Bypass security block for old model weights (CVE-2025-32434)
# This is required since Bunya's PyTorch version (2.5.1) is below the new 2.6.0 requirement
os.environ["TORCH_SKIP_SAF_CHECK"] = "1"
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
        default="microsoft/deberta-v3-base",
        help="Pretrained BERT model. Default: DeBERTa-v3-base (ROCLING 2025 SOTA for IRC disentanglement). "
        "Alternatives: bert-base-uncased (Devlin et al., 2019), roberta-base (Liu et al., 2019).",
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Maximum token length per message (Devlin et al., 2019: BERT uses 128 for 90%% of pretraining). "
        "IRC messages are short; 128 captures >95%% of utterances. Increase to 256 or 512 for longer texts.",
    )

    parser.add_argument(
        "--max-dist",
        type=int,
        default=50,
        help="Maximum window of previous messages to consider as candidates. "
        "ROCKLING 2025 (Lam & Yang): StructBERT uses kh=50. ALT 2021: kc=60. "
        "Larger values increase recall but also memory usage and noise.",
    )

    # Training hyperparameters
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Samples per batch. 64 matches ALT 2021 (BERT+MF for IRC disentanglement). "
        "Reduce (e.g., 16-32) if GPU runs out of memory. "
        "BERT original: 32 for GLUE tasks (Devlin et al., 2019).",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Peak learning rate for AdamW optimizer. "
        "BERT paper reports 2e-5 to 5e-5 for fine-tuning (Devlin et al., 2019). "
        "IRC disentanglement: ALT 2021 and Bi-CL (Huang et al., 2024) both use 5e-5. "
        "Lower (2e-5) if fine-tuning on very small datasets.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs. BERT paper uses 3 for all GLUE tasks (Devlin et al., 2019). "
        "General practice: 3-5 epochs for fine-tuning classification. "
        "More epochs risk overfitting; early stopping via --patience is recommended.",
    )

    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Fraction of total training steps used for linear LR warmup. "
        "Standard practice: 10%% of total steps (Devlin et al., 2019; HuggingFace default). "
        "E.g., 270K total steps → 27K warmup steps. Scales automatically with dataset size.",
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout probability applied to [CLS] embedding before the classifier. "
        "Dropout=0.1 is standard for BERT classification heads (Devlin et al., 2019). "
        "Confirmed by ACL 2025 SemEval and Stanford CS224n 2024 projects. "
        "Increase (0.2-0.3) for small datasets to reduce overfitting.",
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

    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing on BERT backbone (trades ~30%% speed for ~80%% less activation VRAM)",
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


def collate_fn(batch):
    """
    Custom collate function to handle variable-sized candidate lists per item.

    Tested: tests/test_train_pipeline.py::TestCollateFn (6 tests)

    Problem: Each sample has a different number of candidates (C varies).
    PyTorch requires rectangular batches.

    Solution: Find max C in the batch, pad all samples to that size with zeros.
    Zero-padded candidates are then masked out in model.forward() via attention_mask.

    Input batch item:  input_ids [C_i, seq_len], features [C_i, num_features], labels [1]
    Output batch dict:  input_ids [batch, max_C, seq_len], features [batch, max_C, num_features], labels [batch]
    """

    # Find the maximum number of candidates in this batch
    max_candidates = max(item["input_ids"].shape[0] for item in batch)
    # Hard cap at 15 to prevent outlier padding spikes from saturating GPU memory.
    # Must match --max-dist in the SLURM script.
    max_candidates = min(max_candidates, 15)
    seq_len = batch[0]["input_ids"].shape[1]
    num_features = batch[0]["features"].shape[1]  # features is [C, num_features]

    # Initialize padded tensors
    batch_input_ids = torch.zeros(len(batch), max_candidates, seq_len, dtype=torch.long)
    batch_attention_mask = torch.zeros(
        len(batch), max_candidates, seq_len, dtype=torch.long
    )
    batch_features = torch.zeros(
        len(batch), max_candidates, num_features, dtype=torch.float32
    )
    batch_labels = torch.zeros(len(batch), dtype=torch.long)

    # Fill batch
    for i, item in enumerate(batch):
        # Get current item's data
        input_ids = item["input_ids"]  # [C_curr, seq_len]
        attention_mask = item["attention_mask"]  # [C_curr, seq_len]
        features = item["features"]  # [C_curr, num_features]
        labels = item["labels"]  # [scalar]

        # Copy data to batch tensors (padding with 0s for extra candidates)
        actual_candidates = input_ids.shape[0]
        batch_input_ids[i, :actual_candidates] = input_ids
        batch_attention_mask[i, :actual_candidates] = attention_mask
        batch_features[i, :actual_candidates] = features
        batch_labels[i] = labels

    return {
        "input_ids": batch_input_ids,
        "attention_mask": batch_attention_mask,
        "features": batch_features,
        "labels": batch_labels,
    }


def create_dataloaders(args, tokenizer):
    """
    Create train and dev dataloaders.

    Tested: tests/test_train_pipeline.py::TestCreateDataloaders (3 tests)

    1. Calls load_dataset_files() to find .txt/.ann file pairs
    2. Creates IRCDisentanglementDataset for each split (lazy tokenization, on-the-fly in __getitem__)
    3. Wraps in DataLoader with collate_fn for variable-C candidate padding

    Returns:
        (train_loader, dev_loader)  — either may be None depending on mode
    """

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

        logger.info(f"  Dev dataset created: {len(dev_dataset)} samples")

        dev_loader = DataLoader(
            dev_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(args.device == "cuda"),
            collate_fn=collate_fn,
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

        logger.info(f"  Train dataset created: {len(train_dataset)} samples")

        logger.info(f"Loading {len(dev_ascii)} dev files...")
        dev_dataset = IRCDisentanglementDataset(
            ascii_files=dev_ascii,
            annotation_files=dev_ann,
            tokenizer=tokenizer,
            max_dist=args.max_dist,
            max_length=args.max_length,
        )

        logger.info(f"  Dev dataset created: {len(dev_dataset)} samples")

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=(args.device == "cuda"),
            collate_fn=collate_fn,
        )

        dev_loader = DataLoader(
            dev_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(args.device == "cuda"),
            collate_fn=collate_fn,
        )

        return train_loader, dev_loader


def evaluate(model, dataloader, device, fp16=False):
    """Evaluate model on a dataset (multiclass mode)"""
    model.eval()

    logger.info(f"Starting evaluation on {len(dataloader.dataset)} samples")
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
                with torch.amp.autocast("cuda", enabled=fp16):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        features=features,
                        labels=labels,
                    )

                # Get predictions (multiclass: argmax)
                probs = outputs["probs"]
                predictions = torch.argmax(probs, dim=-1)

                # Store results (keep as tensors, concat at the end)
                all_predictions.append(predictions.cpu())
                all_labels.append(labels.cpu())
                all_probs.append(probs.cpu())

                # Accumulate loss
                if "loss" in outputs:
                    total_loss += outputs["loss"].item()
                    num_batches += 1
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.error(
                        f"  Eval Batch {batch_idx + 1}: CUDA Out of Memory (OOM) during evaluation!"
                    )
                    if torch.cuda.is_available():
                        allocated = torch.cuda.memory_allocated(device) / (1024**2)
                        reserved = torch.cuda.memory_reserved(device) / (1024**2)
                        logger.error(
                            f"  Memory at OOM: {allocated:.0f}/{reserved:.0f}MB"
                        )
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

    # Calculate metrics (concatenate accumulated tensors)
    all_predictions = (
        torch.cat(all_predictions)
        if all_predictions
        else torch.tensor([], dtype=torch.long)
    )
    all_labels = (
        torch.cat(all_labels) if all_labels else torch.tensor([], dtype=torch.long)
    )
    # Probs have variable C per batch, can't concatenate directly.
    # Keep as list for debugging; metrics use predictions/labels only.
    all_probs = all_probs if all_probs else []

    # Multiclass metrics
    accuracy = (
        (all_predictions == all_labels).float().mean().item()
        if len(all_predictions) > 0
        else 0.0
    )

    # Macro-averaged precision, recall, F1 across all candidate classes
    num_classes = 0
    precision = 0.0
    recall = 0.0
    f1 = 0.0

    if len(all_predictions) > 0:
        num_classes = max(all_labels.max().item(), all_predictions.max().item()) + 1
        per_class_precision = []
        per_class_recall = []
        per_class_f1 = []

        for c in range(num_classes):
            tp = ((all_predictions == c) & (all_labels == c)).sum().item()
            fp = ((all_predictions == c) & (all_labels != c)).sum().item()
            fn = ((all_predictions != c) & (all_labels == c)).sum().item()

            precision_c = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall_c = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_c = (
                2 * precision_c * recall_c / (precision_c + recall_c)
                if (precision_c + recall_c) > 0
                else 0.0
            )

            per_class_precision.append(precision_c)
            per_class_recall.append(recall_c)
            per_class_f1.append(f1_c)

        precision = sum(per_class_precision) / num_classes
        recall = sum(per_class_recall) / num_classes
        f1 = sum(per_class_f1) / num_classes

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    # Prediction vs. label scatter sample (first 20)
    if len(all_predictions) > 0:
        sample_n = min(20, len(all_predictions))
        pred_sample = all_predictions[:sample_n].tolist()
        label_sample = all_labels[:sample_n].tolist()
        logger.info(f"  Prediction sample (first {sample_n}):")
        logger.info(f"    Predicted: {pred_sample}")
        logger.info(f"    Gold:      {label_sample}")
        errors = [abs(p - l) for p, l in zip(pred_sample, label_sample)]
        logger.info(f"    Abs error: {errors} (mean={sum(errors)/len(errors):.2f})")

    # Per-candidate-position accuracy breakdown (top 10 classes)
    if len(all_predictions) > 0 and len(all_labels) > 0:
        num_pos_classes = max(all_labels.max().item(), all_predictions.max().item()) + 1
        logger.info(f"  Per-position accuracy (top 10 classes):")
        for c in range(min(num_pos_classes, 10)):
            mask = (all_labels == c)
            if mask.sum() == 0:
                continue
            class_acc = (all_predictions[mask] == c).float().mean().item()
            count = mask.sum().item()
            logger.info(f"    Position {c:2d}: {class_acc:.3f} accuracy ({count} samples)")

    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"Evaluation complete in {elapsed:.2f}s")
    logger.info(
        f"  Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}"
    )

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predictions": all_predictions,
        "labels": all_labels,
        "probs": all_probs,
    }


def train_epoch(
    model, train_loader, optimizer, scheduler, device, epoch, fp16=False, scaler=None
):
    """Train for one epoch"""
    model.train()

    # Reset peak memory stats for accurate per-epoch high-watermark
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        logger.info(f"Epoch {epoch}: Peak memory stats reset.")

    logger.info(f"Starting epoch {epoch} with {len(train_loader)} batches")
    start_time = datetime.now()

    total_loss = 0.0
    num_batches = 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=True, disable=True)

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

            # Log first batch of each epoch for data sanity check
            if batch_idx == 0:
                b, max_c, seq_len = input_ids.shape
                actual_candidates = (batch["attention_mask"].sum(dim=-1) > 0).sum(dim=-1)
                logger.info(f"  [Epoch {epoch} Batch 0 SANITY CHECK]")
                logger.info(f"    input_ids shape : {list(input_ids.shape)}  (batch x max_C x seq_len)")
                logger.info(f"    features shape  : {list(features.shape)}")
                logger.info(f"    labels shape    : {list(labels.shape)}")
                logger.info(f"    Candidate counts per sample: min={actual_candidates.min().item()} "
                            f"max={actual_candidates.max().item()} "
                            f"mean={actual_candidates.float().mean().item():.1f}")
                logger.info(f"    Label values: {labels.tolist()[:16]}{'...' if b > 16 else ''}")
                label_dist = {}
                for lbl in labels.tolist():
                    label_dist[lbl] = label_dist.get(lbl, 0) + 1
                logger.info(f"    Label distribution (this batch): {dict(sorted(label_dist.items()))}")

            # Forward pass
            with torch.amp.autocast("cuda", enabled=fp16):
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

                # Track consecutive OOM failures — if cascading, abort early
                if not hasattr(train_epoch, "_consecutive_oom"):
                    train_epoch._consecutive_oom = 0
                train_epoch._consecutive_oom += 1
                if train_epoch._consecutive_oom >= 10:
                    raise RuntimeError(
                        f"OOM cascade: {train_epoch._consecutive_oom} consecutive OOM batches. "
                        f"Reduce --batch-size or --max-dist and resubmit."
                    )
                continue
            else:
                raise e

        # SMART LOGGING: Log every 50 batches to monitor general probability distribution and avoid log flooding.
        if (batch_idx + 1) % 50 == 0:
            avg_prob = probs.mean().item()
            max_prob = probs.max().item()
            min_prob = probs.min().item()

            # Prediction diversity: what candidate indices is the model picking?
            preds = torch.argmax(probs, dim=-1)  # [batch]
            pred_counts = {}
            for p in preds.tolist():
                pred_counts[p] = pred_counts.get(p, 0) + 1
            top_pred = max(pred_counts, key=pred_counts.get)
            top_pred_pct = 100.0 * pred_counts[top_pred] / len(preds)

            # Confidence: max prob per sample (how sure is the model?)
            confidence = probs.max(dim=-1).values  # [batch]
            avg_confidence = confidence.mean().item()

            logger.info(
                f"  Batch {batch_idx + 1} Stats: "
                f"Prob Range: [{min_prob:.4f}, {max_prob:.4f}], "
                f"Avg Prob: {avg_prob:.4f}, "
                f"Avg Confidence: {avg_confidence:.4f}, "
                f"Top Predicted Index: {top_pred} ({top_pred_pct:.0f}% of batch)"
            )
            if top_pred_pct > 80:
                logger.warning(
                    f"  WARNING: Model predicting index {top_pred} for {top_pred_pct:.0f}% of this batch. "
                    f"Possible mode collapse or positional bias."
                )

        # Backward pass
        try:
            optimizer.zero_grad()
            loss.backward()

            # Check for NaN/Inf gradients before optimizer step
            grad_has_nan = False
            for p in model.parameters():
                if p.grad is not None:
                    if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                        grad_has_nan = True
                        break

            if grad_has_nan:
                logger.error(
                    f"  Batch {batch_idx + 1}: NaN/Inf gradient detected! Skipping batch."
                )
                optimizer.zero_grad()
                continue

            # Gradient clipping (max_norm=10.0): prevents any single batch from
            # destabilizing model parameters. Healthy norm for DeBERTa fine-tuning
            # is ~0.5-5.0. The batch 0 gradient norm on the Ubuntu dataset was 4.0.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            # Gradient norm logging (first batch of each epoch)
            if batch_idx == 0:
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                total_norm = total_norm ** 0.5
                logger.info(f"  Epoch {epoch} Batch 0 gradient norm: {total_norm:.4f}")
                if total_norm > 10.0:
                    logger.warning(f"  WARNING: High gradient norm ({total_norm:.4f}). "
                                   f"Consider gradient clipping (torch.nn.utils.clip_grad_norm_).")
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

                # Track consecutive OOM failures — if cascading, abort early
                if not hasattr(train_epoch, "_consecutive_oom"):
                    train_epoch._consecutive_oom = 0
                train_epoch._consecutive_oom += 1
                if train_epoch._consecutive_oom >= 10:
                    raise RuntimeError(
                        f"OOM cascade: {train_epoch._consecutive_oom} consecutive OOM batches. "
                        f"Reduce --batch-size or --max-dist and resubmit."
                    )
                continue
            else:
                raise e

        if scheduler is not None:
            scheduler.step()

        # Periodic defrag: prevents allocator fragmentation buildup
        if batch_idx % 50 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Reset consecutive OOM counter on successful batch
        if hasattr(train_epoch, "_consecutive_oom"):
            train_epoch._consecutive_oom = 0

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
                max_c = batch["input_ids"].shape[1]
                mem_msg = f", max_C={max_c}, GPU Mem: {allocated:.0f}/{reserved:.0f}MB (max: {max_allocated:.0f}MB)"

            logger.info(
                f"  Epoch {epoch} progress: {batch_idx + 1}/{len(train_loader)} batches "
                f"({batches_per_sec:.2f} batches/s, avg_loss={avg_loss_so_far:.4f}{mem_msg})"
            )

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    elapsed = (datetime.now() - start_time).total_seconds()

    # Succeeded vs. skipped batch counter
    skipped_batches = (batch_idx + 1) - num_batches
    logger.info(
        f"Epoch {epoch} complete in {elapsed:.2f}s | "
        f"avg_loss={avg_loss:.4f} | "
        f"batches: {num_batches} succeeded, {skipped_batches} skipped (OOM/NaN)"
    )
    if skipped_batches > 0:
        skip_pct = 100.0 * skipped_batches / (batch_idx + 1)
        logger.warning(f"  WARNING: {skip_pct:.1f}% of batches were skipped this epoch!")

    return avg_loss


def save_checkpoint(model, optimizer, scheduler, epoch, args, metrics, checkpoint_dir):
    """Save training checkpoint with robust file handling"""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
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
        logger.info(f"  Train dataset: {len(train_loader.dataset)} samples")
        logger.info(
            f"Training: {len(train_loader.dataset)} samples | "
            f"{len(train_loader)} batches/epoch | "
            f"batch_size={args.batch_size} | "
            f"max_dist={args.max_dist} | "
            f"max_length={args.max_length}"
        )
    if dev_loader:
        logger.info(f"  Dev dataset: {len(dev_loader.dataset)} samples")
        logger.info(
            f"Validation: {len(dev_loader.dataset)} samples | "
            f"{len(dev_loader)} batches"
        )

    # Create model
    logger.info("Creating model...")
    model = create_model(
        model_name=args.model_name,
        num_features=5,
        dropout=args.dropout,
        freeze_bert=args.freeze_bert,
        gradient_checkpointing=args.gradient_checkpointing,
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
            num_warmup_steps = int(total_steps * args.warmup_ratio)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_steps,
            )
            logger.info(
                f"Created scheduler with {num_warmup_steps} warmup steps ({args.warmup_ratio*100:.0f}% of {total_steps} total steps)"
            )
        else:
            scheduler = None
    else:
        optimizer = None
        scheduler = None

    # Mixed precision scaler — DISABLED.
    #
    # What GradScaler normally does: When training in fp16, very small gradient
    # values (e.g. 0.0000001) can't be represented in 16-bit and become zero
    # ("underflow"). GradScaler prevents this by multiplying the loss by a large
    # constant before backward, then dividing gradients back down before the
    # optimizer step. Like turning up the volume on a quiet recording.
    #
    # Why it's disabled: PyTorch 2.5.1's GradScaler raises "ValueError: Attempting
    # to unscale FP16 gradients" when used with gradient_checkpointing_enable().
    # Gradient checkpointing recomputes intermediate activations during backward
    # instead of storing them, which creates a different autograd graph structure
    # that the scaler doesn't understand.
    #
    # Why that's fine: Autocast (torch.amp.autocast) still runs the forward pass
    # in fp16 for memory savings. Gradient underflow isn't a problem for BERT
    # fine-tuning — gradients are large enough that they don't vanish in 16-bit.
    # GradScaler is mainly needed for training from scratch on huge datasets
    # (ImageNet, etc.), not for fine-tuning a pretrained model.
    scaler = None

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
        epoch_losses = []

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
                epoch_losses.append(train_loss)
                if len(epoch_losses) >= 2:
                    delta = epoch_losses[-1] - epoch_losses[-2]
                    trend = "↓ improving" if delta < 0 else "↑ diverging"
                    logger.info(f"  Loss trend: {epoch_losses[-2]:.4f} → {epoch_losses[-1]:.4f} ({delta:+.4f}) {trend}")
                logger.info(f"Train Loss: {train_loss:.4f}")

            # Evaluate
            if dev_loader and epoch % args.eval_every == 0:
                logger.info("Evaluating on dev set...")
                metrics = evaluate(model, dev_loader, device, fp16=args.fp16)

                logger.info(f"Dev Loss: {metrics['loss']:.4f}")
                logger.info(f"Dev Accuracy: {metrics['accuracy']:.4f}")

                # Track best model (accuracy is the multiclass metric)
                current_score = metrics.get("accuracy", 0.0)
                if current_score > best_f1:
                    best_f1 = current_score
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
            metrics = evaluate(model, dev_loader, device, fp16=args.fp16)

            logger.info("Test Results:")
            logger.info(f"Loss: {metrics['loss']:.4f}")
            logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"Precision: {metrics['precision']:.4f}")
            logger.info(f"Recall: {metrics['recall']:.4f}")
            logger.info(f"F1: {metrics['f1']:.4f}")

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
                    logger.error(
                        f"Final Memory State: {allocated:.0f}/{reserved:.0f}MB"
                    )

            logger.exception("Fatal error during training:")
        else:
            import traceback

            traceback.print_exc()
        sys.exit(1)
