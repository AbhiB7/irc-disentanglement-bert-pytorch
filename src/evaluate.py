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
    
    parser.add_argument(
        "--metrics",
        type=str,
        default="both",
        choices=["pairwise", "clustering", "both"],
        help="Which metrics to compute: pairwise (per-message), clustering (VI/ARI), or both",
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


def load_dataset(data_dir, split, max_dist, max_length, batch_size, num_workers, device, tokenizer):
    """Load the dataset for evaluation (dev or test split)"""
    ascii_files, ann_files = load_dataset_files(data_dir, split=split)
    
    dataset = IRCDisentanglementDataset(
        ascii_files=ascii_files,
        annotation_files=ann_files,
        tokenizer=tokenizer,
        max_dist=max_dist,
        max_length=max_length,
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        collate_fn=collate_fn,
    )
    
    return loader


def load_gold_clusters(data_dir, split):
    """
    Load gold clusters from gold.{split}.clusters.txt.
    
    Format: "conversation_name:msg_idx msg_idx msg_idx ..."
    Each line = one cluster (thread) with all message indices belonging to it.
    
    Returns:
        all_clusters: dict mapping conversation_name -> list of sets of message indices
    """
    clusters_path = Path(data_dir) / f"gold.{split}.clusters.txt"
    if not clusters_path.exists():
        logger.warning(f"Gold clusters file not found: {clusters_path}")
        return {}
    
    all_clusters = {}
    with open(clusters_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            # First part: "conv_name:msg_idx" e.g. "2004-11-15_03:1018"
            # Rest: message indices
            conv_msg = parts[0]
            if ":" not in conv_msg:
                continue
            conv_name = conv_msg.split(":")[0]
            msg_indices = [int(conv_msg.split(":")[1])] + [int(p) for p in parts[1:]]
            all_clusters.setdefault(conv_name, []).append(set(msg_indices))
    
    logger.info(f"  Loaded {sum(len(v) for v in all_clusters.values())} clusters across {len(all_clusters)} conversations")
    return all_clusters


def compute_ari_and_vi(gold_clusters, pred_clusters):
    """
    Compute Adjusted Rand Index (ARI) and Variation of Information (VI)
    between gold and predicted clusterings.
    
    Args:
        gold_clusters: list of sets of message indices
        pred_clusters: list of sets of message indices
    
    Returns:
        (ari, vi) tuple
    """
    import math
    
    # Get all messages
    all_messages = set()
    for c in gold_clusters:
        all_messages.update(c)
    for c in pred_clusters:
        all_messages.update(c)
    
    n = len(all_messages)
    if n == 0:
        return 0.0, 0.0
    
    # Build message-to-cluster mapping
    gold_assignment = {}  # msg -> cluster_id
    for cid, cluster in enumerate(gold_clusters):
        for msg in cluster:
            gold_assignment[msg] = cid
    
    pred_assignment = {}
    for cid, cluster in enumerate(pred_clusters):
        for msg in cluster:
            pred_assignment[msg] = cid
    
    n_gold = len(gold_clusters)
    n_pred = len(pred_clusters)
    
    # Build contingency table
    contingency = {}
    for msg in all_messages:
        g = gold_assignment.get(msg, -1)
        p = pred_assignment.get(msg, -1)
        contingency[(g, p)] = contingency.get((g, p), 0) + 1
    
    # ---- ARI Calculation ----
    # Sum over rows and columns
    row_sums = {}
    col_sums = {}
    total_pairs = n * (n - 1) / 2
    
    for (g, p), count in contingency.items():
        row_sums[g] = row_sums.get(g, 0) + count
        col_sums[p] = col_sums.get(p, 0) + count
    
    # Sum of pairs in agreement
    sum_nij_choose2 = sum(
        count * (count - 1) / 2 for count in contingency.values()
    )
    sum_ai_choose2 = sum(
        count * (count - 1) / 2 for count in row_sums.values()
    )
    sum_bj_choose2 = sum(
        count * (count - 1) / 2 for count in col_sums.values()
    )
    
    # Expected index (under hypergeometric assumption)
    expected = sum_ai_choose2 * sum_bj_choose2 / total_pairs if total_pairs > 0 else 0.0
    max_index = (sum_ai_choose2 + sum_bj_choose2) / 2
    
    ari = (sum_nij_choose2 - expected) / (max_index - expected) if (max_index - expected) != 0 else 0.0
    
    # ---- VI Calculation ----
    # Mutual Information: MI(X,Y) = sum_x sum_y p(x,y) * log(p(x,y) / (p(x)*p(y)))
    mi = 0.0
    for (g, p), count in contingency.items():
        if count > 0:
            pxy = count / n
            px = row_sums.get(g, 0) / n
            py = col_sums.get(p, 0) / n
            if px > 0 and py > 0:
                mi += pxy * math.log(pxy / (px * py), 2) if pxy > 0 else 0.0
    
    # Entropy of gold: H(X) = -sum p(x) * log(p(x))
    h_gold = 0.0
    for count in row_sums.values():
        px = count / n
        if px > 0:
            h_gold -= px * math.log(px, 2)
    
    # Entropy of pred: H(Y) = -sum p(y) * log(p(y))
    h_pred = 0.0
    for count in col_sums.values():
        py = count / n
        if py > 0:
            h_pred -= py * math.log(py, 2)
    
    # VI = H(X|Y) + H(Y|X) = H(X) + H(Y) - 2*MI
    vi = h_gold + h_pred - 2 * mi
    
    return ari, vi


def cluster_from_predictions(loader, predictions, dataset):
    """
    Build predicted clusters from pairwise parent-child predictions.
    
    Each sample in the dataset has:
    - conversation_map: (conv_idx, child_msg_idx, candidate_indices)
      where candidate_indices = [(conv_idx, i, j), ...]
    - prediction: index into candidate_indices
    
    For each prediction, the predicted parent is candidate_indices[pred][2] = parent_msg_idx.
    
    Returns:
        per_conversation_clusters: dict mapping conversation_name -> list of sets
    """
    from collections import defaultdict
    
    # Build predicted edges: for each sample, child -> predicted parent
    # Also need gold edges from the annotation files
    edges = defaultdict(list)  # conv_name -> [(child, parent)]
    
    for idx, pred in enumerate(predictions.tolist()):
        conv_idx, child_idx, candidate_indices = dataset.conversation_map[idx]
        
        if pred < 0 or pred >= len(candidate_indices):
            continue  # Skip invalid predictions (shouldn't happen)
        
        parent_idx = candidate_indices[pred][2]  # j = parent message index
        conv_name = dataset.conversations[conv_idx].name
        edges[conv_name].append((child_idx, parent_idx))
    
    # Build predicted clusters via connected components
    per_conversation_clusters = {}
    
    for conv_name, edge_list in edges.items():
        # Build adjacency list
        nodes = set()
        adj = defaultdict(set)
        for child, parent in edge_list:
            nodes.add(child)
            nodes.add(parent)
            adj[child].add(parent)
            adj[parent].add(child)
        
        # BFS to find connected components
        visited = set()
        clusters = []
        for node in nodes:
            if node not in visited:
                cluster = set()
                stack = [node]
                while stack:
                    curr = stack.pop()
                    if curr not in visited:
                        visited.add(curr)
                        cluster.add(curr)
                        stack.extend(adj[curr] - visited)
                if cluster:
                    clusters.append(cluster)
        
        per_conversation_clusters[conv_name] = clusters
    
    return per_conversation_clusters


def compute_clustering_eval(loader, predictions, dataset, gold_clusters):
    """
    Compute clustering metrics comparing predicted clusters vs gold clusters.
    
    Returns:
        metrics dict with ari, vi, and per-conversation breakdown
    """
    pred_clusters = cluster_from_predictions(loader, predictions, dataset)
    
    total_ari = 0.0
    total_vi = 0.0
    n_convs = 0
    
    per_conv_results = []
    
    for conv_name, gold_conv_clusters in gold_clusters.items():
        if conv_name not in pred_clusters:
            continue
        
        pred_conv_clusters = pred_clusters[conv_name]
        ari, vi = compute_ari_and_vi(gold_conv_clusters, pred_conv_clusters)
        
        total_ari += ari
        total_vi += vi
        n_convs += 1
        
        per_conv_results.append((conv_name, ari, vi, len(gold_conv_clusters), len(pred_conv_clusters)))
    
    # Average across conversations
    avg_ari = total_ari / n_convs if n_convs > 0 else 0.0
    avg_vi = total_vi / n_convs if n_convs > 0 else 0.0
    
    # Log per-conversation results
    logger.info(f"  Per-conversation clustering metrics ({n_convs} conversations):")
    for conv_name, ari, vi, n_gold, n_pred in sorted(per_conv_results, key=lambda x: -x[1])[:10]:
        logger.info(f"    {conv_name}: ARI={ari:.4f}, VI={vi:.4f} ({n_gold} gold clusters, {n_pred} pred clusters)")
    if len(per_conv_results) > 10:
        logger.info(f"    ... ({len(per_conv_results) - 10} more conversations)")
    
    return {
        "clustering_ari": avg_ari,
        "clustering_vi": avg_vi,
        "num_conversations": n_convs,
    }


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
    loader = load_dataset(
        args.data_dir,
        args.split,
        args.max_dist,
        args.max_length,
        args.batch_size,
        args.num_workers,
        device,
        tokenizer,
    )
    logger.info(f"{args.split} dataset: {len(loader.dataset)} samples")
    
    # Run evaluation
    if args.metrics in ["pairwise", "both"]:
        logger.info("=== Pairwise Metrics (Per-Message Accuracy) ===")
        metrics = evaluate(model, loader, device)
        
        logger.info("=" * 80)
        logger.info("Evaluation Results (Multiclass Mode)")
        logger.info("=" * 80)
        logger.info(f"Loss: {metrics['loss']:.4f}")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1: {metrics['f1']:.4f}")
        logger.info("=" * 80)
    
    if args.metrics in ["clustering", "both"]:
        logger.info("=== Clustering Metrics (VI / ARI) ===")
        
        # Compute clustering metrics
        gold_clusters = load_gold_clusters(args.data_dir, args.split)
        if gold_clusters:
            # Need to get predictions first - run evaluate if not done yet
            if args.metrics == "clustering":
                metrics = evaluate(model, loader, device)
            
            clustering_metrics = compute_clustering_eval(
                loader, metrics["predictions"], loader.dataset, gold_clusters
            )
            
            logger.info("=" * 80)
            logger.info("Clustering Evaluation Results")
            logger.info("=" * 80)
            logger.info(f"ARI (Adjusted Rand Index): {clustering_metrics['clustering_ari']:.4f}")
            logger.info(f"VI  (Variation of Information): {clustering_metrics['clustering_vi']:.4f}")
            logger.info(f"Conversations evaluated: {clustering_metrics['num_conversations']}")
            logger.info("=" * 80)
    else:
        logger.info("Skipping clustering metrics (gold clusters file may be missing)")
    
    logger.info(f"Results saved to: {LOG_FILE}")


if __name__ == "__main__":
    main()