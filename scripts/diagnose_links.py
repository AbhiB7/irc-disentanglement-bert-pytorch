"""Diagnose: count self-links vs cross-links and their distances."""
import glob, statistics

data_dir = "data"

self_link_count = 0
cross_link_count = 0
cross_link_distances = []
min_annotated_idx = float('inf')
max_annotated_idx = float('-inf')

# Only analyze the main Ubuntu IRC train/dev/test annotation files
# (skip channel-two and annotation-process which have different formats)
for f in sorted(glob.glob(f"{data_dir}/train/*.annotation*", recursive=False)):
    with open(f) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                # Handle colon-prefixed indices like ":1000"
                child_str = parts[0].lstrip(":")
                parent_str = parts[1].lstrip(":")
                child, parent = int(child_str), int(parent_str)
                min_annotated_idx = min(min_annotated_idx, child, parent)
                max_annotated_idx = max(max_annotated_idx, child, parent)
                if child == parent:
                    self_link_count += 1
                else:
                    cross_link_count += 1
                    cross_link_distances.append(abs(child - parent))

total = self_link_count + cross_link_count
print(f"Total gold links: {total}")
print(f"Self-links: {self_link_count} ({100*self_link_count/total:.1f}%)")
print(f"Cross-links: {cross_link_count} ({100*cross_link_count/total:.1f}%)")
print(f"Annotated message range: {min_annotated_idx} to {max_annotated_idx}")
print()
if cross_link_distances:
    print(f"Cross-link distance stats:")
    print(f"  Median: {statistics.median(cross_link_distances)}")
    print(f"  Mean:   {statistics.mean(cross_link_distances):.1f}")
    print(f"  Max:    {max(cross_link_distances)}")
    print(f"  Min:    {min(cross_link_distances)}")
    print(f"  Within 15: {sum(1 for d in cross_link_distances if d <= 15)} ({100*sum(1 for d in cross_link_distances if d <= 15)/len(cross_link_distances):.1f}%)")
    print(f"  Within 50: {sum(1 for d in cross_link_distances if d <= 50)} ({100*sum(1 for d in cross_link_distances if d <= 50)/len(cross_link_distances):.1f}%)")
    print(f"  Within 100: {sum(1 for d in cross_link_distances if d <= 100)} ({100*sum(1 for d in cross_link_distances if d <= 100)/len(cross_link_distances):.1f}%)")