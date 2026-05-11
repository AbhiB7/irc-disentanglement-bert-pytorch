"""Analyze gold cluster files to count singletons (clusters of size 1).

This helps explain why ARI=0: if most gold clusters are singletons,
the ARI denominator collapses and the metric becomes undefined.

Usage:
    python scripts/analyze_gold_clusters.py
"""

from pathlib import Path


def analyze_gold_clusters(filepath):
    """Read a gold clusters file and report statistics."""
    clusters = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(":")
            conv_name = parts[0]
            msg_indices = [int(x) for x in parts[1].split()]
            clusters.append((conv_name, msg_indices))

    total_clusters = len(clusters)
    singletons = [c for c in clusters if len(c[1]) == 1]
    non_singletons = [c for c in clusters if len(c[1]) > 1]

    print(f"File: {filepath}")
    print(f"  Total clusters: {total_clusters}")
    print(f"  Singleton clusters (size=1): {len(singletons)} ({100 * len(singletons) / total_clusters:.1f}%)")
    print(f"  Non-singleton clusters (size>1): {len(non_singletons)} ({100 * len(non_singletons) / total_clusters:.1f}%)")

    if non_singletons:
        sizes = [len(c[1]) for c in non_singletons]
        print(f"  Non-singleton cluster sizes: min={min(sizes)}, max={max(sizes)}, mean={sum(sizes) / len(sizes):.1f}")

    # Print breakdown by conversation
    print(f"\n  Per-conversation breakdown:")
    convs = {}
    for conv_name, msgs in clusters:
        if conv_name not in convs:
            convs[conv_name] = {"total": 0, "singletons": 0}
        convs[conv_name]["total"] += 1
        if len(msgs) == 1:
            convs[conv_name]["singletons"] += 1

    for conv_name, stats in sorted(convs.items()):
        pct = 100.0 * stats["singletons"] / stats["total"]
        print(f"    {conv_name}: {stats['singletons']}/{stats['total']} clusters are singletons ({pct:.1f}%)")

    return total_clusters, len(singletons)


if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent / "data"

    for filename in ["gold.dev.clusters.txt", "gold.test.clusters.txt"]:
        analyze_gold_clusters(data_dir / filename)
        print()