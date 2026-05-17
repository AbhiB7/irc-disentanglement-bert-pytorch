"""
Generate synthetic IRC conversations with interleaved thread structure.

This version creates conversations where multiple threads run INTERLEAVED (like real IRC),
not sequentially. The correct parent is often 2-5 messages back, not the immediately
previous message. This tests whether the model can learn content-based patterns
rather than relying on recency bias.

Usage: python scripts/generate_synthetic_data.py --output-dir data/synthetic_interleaved --num-conversations 5
"""

import argparse
import random
from pathlib import Path
from datetime import datetime, timedelta


# Thread templates with clear topic separation
THREAD_TEMPLATES = {
    "python_help": {
        "speakers": ["Alice", "Bob"],
        "messages": [
            "How do I install {package}?",
            "Use pip install {package}.",
            "What about CUDA support?",
            "You need to install the cuda version specifically.",
            "Getting an error: {error}",
            "Try downgrading to version {version}.",
        ],
    },
    "weather": {
        "speakers": ["Charlie", "Diana"],
        "messages": [
            "Weather today?",
            "Sunny, {temp}°C.",
            "Will it rain tomorrow?",
            "No, clear skies expected.",
            "What about the weekend?",
            "Saturday might have showers.",
        ],
    },
    "food": {
        "speakers": ["Eve", "Frank"],
        "messages": [
            "Lunch recommendations?",
            "Try the {cuisine} place on {street}.",
            "Is it expensive?",
            "Around ${price} per person.",
            "Do I need reservations?",
            "Yes, it's quite popular.",
        ],
    },
    "gaming": {
        "speakers": ["Grace", "Henry"],
        "messages": [
            "Anyone playing {game}?",
            "Yeah, just finished {level}.",
            "How long did it take?",
            "About {hours} hours, pretty short.",
            "Worth buying?",
            "Definitely, 9/10 rating.",
        ],
    },
}


def generate_interleaved_conversation(conv_id, num_threads=3, msgs_per_thread=6):
    """
    Generate one conversation with interleaved threads.
    
    Threads are interleaved so messages from different threads are mixed,
    simulating real IRC behavior. The correct parent for a message is often
    not the immediately previous message, but an earlier message in the same thread.
    
    Returns: (messages, annotations, next_msg_idx)
    """
    messages = []
    annotations = []  # (child, parent)
    
    start_time = datetime(2024, 1, 1, 10, 0)  # 10:00 AM
    msg_idx = 1000 + conv_id * 100  # Unique starting index
    
    # Select threads
    thread_topics = random.sample(list(THREAD_TEMPLATES.keys()), min(num_threads, len(THREAD_TEMPLATES)))
    
    # Initialize thread state
    thread_states = {}
    for topic in thread_topics:
        thread_states[topic] = {
            "messages": [],  # List of (msg_idx, speaker, text)
            "template": THREAD_TEMPLATES[topic],
            "next_msg_pos": 0,  # Next message position in template
        }
    
    # Interleave messages from different threads
    # We'll round-robin through threads until all threads have emitted all messages
    thread_order = list(thread_topics)
    random.shuffle(thread_order)  # Randomize initial order
    
    thread_idx = 0
    max_messages = num_threads * msgs_per_thread
    
    for _ in range(max_messages):
        # Pick next thread (round-robin)
        topic = thread_order[thread_idx % len(thread_order)]
        state = thread_states[topic]
        
        # Skip if this thread is exhausted
        if state["next_msg_pos"] >= len(state["template"]["messages"]):
            thread_idx += 1
            continue
        
        # Get message template
        msg_template = state["template"]["messages"][state["next_msg_pos"]]
        speaker = state["template"]["speakers"][state["next_msg_pos"] % 2]
        
        # Format message
        text = msg_template.format(
            package="PyTorch",
            error="CUDA out of memory",
            version="2.0.1",
            temp=random.randint(20, 35),
            cuisine=random.choice(["Italian", "Mexican", "Thai"]),
            street=random.choice(["Main St", "Oak Ave", "5th St"]),
            price=random.randint(15, 40),
            game="Baldur's Gate 3",
            level="Act 2",
            hours=random.randint(20, 60),
        )
        
        # Timestamp
        timestamp = start_time + timedelta(minutes=len(messages) * 2)
        time_str = f"[{timestamp.hour:02d}:{timestamp.minute:02d}]"
        
        # Format IRC line
        irc_line = f"{time_str} <{speaker}> {text}"
        messages.append(irc_line)
        
        # Create annotation
        if state["next_msg_pos"] == 0:
            # First message in thread: it's a root (no parent)
            # Don't add any annotation - roots have no gold parent
            pass
        else:
            # Link to previous message in SAME thread
            prev_msg_idx = state["messages"][-1][0]  # (msg_idx, speaker, text) of previous in thread
            annotations.append((msg_idx, prev_msg_idx))
        
        # Record this message in thread state
        state["messages"].append((msg_idx, speaker, text))
        state["next_msg_pos"] += 1
        
        msg_idx += 1
        thread_idx += 1
    
    return messages, annotations, msg_idx


def write_conversation(output_dir, conv_id, messages, annotations):
    """Write .ascii.txt and .annotation.txt files"""
    ascii_path = output_dir / f"{conv_id}.ascii.txt"
    ann_path = output_dir / f"{conv_id}.annotation.txt"
    
    # Write ASCII file
    with open(ascii_path, "w") as f:
        f.write("\n".join(messages) + "\n")
    
    # Write annotation file (child parent pairs)
    with open(ann_path, "w") as f:
        for child, parent in annotations:
            f.write(f"{child} {parent}\n")
    
    return ascii_path, ann_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="data/synthetic_interleaved", help="Output directory")
    parser.add_argument("--num-conversations", type=int, default=5, help="Number of conversations")
    parser.add_argument("--num-threads", type=int, default=3, help="Threads per conversation")
    parser.add_argument("--msgs-per-thread", type=int, default=6, help="Messages per thread")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {args.num_conversations} synthetic conversations with interleaved threads...")
    
    for conv_id in range(args.num_conversations):
        messages, annotations, _ = generate_interleaved_conversation(
            conv_id, args.num_threads, args.msgs_per_thread
        )
        ascii_path, ann_path = write_conversation(output_dir, conv_id, messages, annotations)
        print(f"  Written: {ascii_path.name}, {ann_path.name} ({len(messages)} messages, {len(annotations)} annotations)")
    
    print(f"\nDone! Synthetic data in: {output_dir}")
    print(f"Threads are interleaved - the correct parent is often NOT the immediately previous message.")


if __name__ == "__main__":
    main()