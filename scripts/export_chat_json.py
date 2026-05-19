#!/usr/bin/env python3
"""
Export IRC conversation data to JSON for visualization.

Usage:
    python scripts/export_chat_json.py \
        --ascii data/tiny/dev/tiny.dev.ascii.txt \
        --annotation data/tiny/dev/tiny.dev.annotation.txt \
        --output app/data/tiny.dev.json
"""

import argparse
import json
import os
import re


def parse_irc_line(line):
    """Parse IRC line into (timestamp, speaker, text, is_system)."""
    line = line.strip()
    if line.startswith("==="):
        return None, "SYSTEM", line, True
    m = re.match(r"^\[(\d{2}):(\d{2})\] <([^>]+)> (.*)$", line)
    if m:
        return f"{m.group(1)}:{m.group(2)}", m.group(3), m.group(4), False
    return None, "UNKNOWN", line, True


def load_annotations(path):
    """Load annotation file. Returns dict child -> parent."""
    links = {}
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0] != "-":
                parent = int(parts[0])
                child = int(parts[1])
                links[child] = parent
    return links


def compute_threads(messages, links):
    """Union-Find over messages. Returns dict root -> [msg_indices]."""
    n = len(messages)
    uf = list(range(n))

    def find(x):
        while uf[x] != x:
            uf[x] = uf[uf[x]]
            x = uf[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            uf[rb] = ra

    for link in links:
        c = link["child"]
        p = link["parent"]
        if c != p:
            union(c, p)

    buckets = {}
    for i in range(n):
        root = find(i)
        buckets.setdefault(root, []).append(i)
    return buckets


def main():
    parser = argparse.ArgumentParser(
        description="Export IRC conversation data to JSON"
    )
    parser.add_argument("--ascii", required=True)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    messages = []
    with open(args.ascii, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            ts, speaker, text, is_sys = parse_irc_line(line)
            messages.append({
                "index": idx,
                "timestamp": ts,
                "speaker": speaker,
                "text": text,
                "is_system": is_sys,
            })

    annotations = load_annotations(args.annotation)

    links = []
    for child, parent in annotations.items():
        links.append({"child": child, "parent": parent})

    thread_dict = compute_threads(messages, links)
    thread_list = []
    for tid, indices in thread_dict.items():
        thread_list.append({
            "id": tid,
            "messages": sorted(indices),
            "size": len(indices),
        })

    name = os.path.basename(args.ascii)
    for ext in [".ascii.txt", ".annotation.txt", ".raw.txt", ".tok.txt"]:
        if name.endswith(ext):
            name = name[:-len(ext)]
            break

    output = {
        "name": name,
        "messages": messages,
        "links": links,
        "threads": sorted(thread_list, key=lambda t: t["size"], reverse=True),
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(
        f"Exported {len(messages)} messages, "
        f"{len(links)} links, {len(thread_list)} threads "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()