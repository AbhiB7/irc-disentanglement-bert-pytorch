"""
Generate synthetic IRC conversations with clear thread structure.

Usage: python scripts/generate_synthetic_data.py --output-dir data/synthetic --num-conversations 5
"""

import argparse
import random
from pathlib import Path
from datetime import datetime, timedelta


# Thread templates with clear topic separation
THREAD_TEMPLATES = {
    "python_help": [
        "How do I install {package}?",
        "Use pip install {package}.",
        "What about CUDA support?",
        "You need to install the cuda version specifically.",
        "Getting an error: {error}",
        "Try downgrading to version {version}.",
    ],
    "weather": [
        "Weather today?",
        "Sunny, {temp}°C.",
        "Will it rain tomorrow?",
        "No, clear skies expected.",
        "What about the weekend?",
        "Saturday might have showers.",
    ],
    "food": [
        "Lunch recommendations?",
        "Try the {cuisine} place on {street}.",
        "Is it expensive?",
        "Around ${price} per person.",
        "Do I need reservations?",
        "Yes, it's quite popular.",
    ],
    "gaming": [
        "Anyone playing {game}?",
        "Yeah, just finished {level}.",
        "How long did it take?",
        "About {hours} hours, pretty short.",
        "Worth buying?",
        "Definitely, 9/10 rating.",
    ],
}


def generate_conversation(conv_id, num_threads=3, msgs_per_thread=6):
    """Generate one conversation with clear thread structure"""
    messages = []
    annotations = []  # (child, parent)
    
    start_time = datetime(2024, 1, 1, 10, 0)  # 10:00 AM
    msg_idx = 1000 + conv_id * 100  # Unique starting index
    
    thread_topics = random.sample(list(THREAD_TEMPLATES.keys()), min(num_threads, len(THREAD_TEMPLATES)))
    
    for thread_id, topic in enumerate(thread_topics):
        template = THREAD_TEMPLATES[topic]
        thread_start_idx = len(messages)
        
        for i, msg_template in enumerate(template[:msgs_per_thread]):
            # Format message
            speaker = "Alice" if i % 2 == 0 else "Bob"
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
            
            # Annotation: link to previous message in thread (not always the immediate previous!)
            if i == 0:
                # First message in thread: link to itself (root)
                annotations.append((msg_idx, msg_idx))
            else:
                # Link to previous message in SAME thread (not necessarily msg_idx-1)
                parent_idx = msg_idx - 1
                annotations.append((msg_idx, parent_idx))
            
            msg_idx += 1
    
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
    parser.add_argument("--output-dir", type=str, default="data/synthetic", help="Output directory")
    parser.add_argument("--num-conversations", type=int, default=5, help="Number of conversations")
    parser.add_argument("--num-threads", type=int, default=3, help="Threads per conversation")
    parser.add_argument("--msgs-per-thread", type=int, default=6, help="Messages per thread")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {args.num_conversations} synthetic conversations...")
    
    for conv_id in range(args.num_conversations):
        messages, annotations, _ = generate_conversation(
            conv_id, args.num_threads, args.msgs_per_thread
        )
        ascii_path, ann_path = write_conversation(output_dir, conv_id, messages, annotations)
        print(f"  Written: {ascii_path.name}, {ann_path.name} ({len(messages)} messages)")
    
    print(f"\nDone! Synthetic data in: {output_dir}")


if __name__ == "__main__":
    main()