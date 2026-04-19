import os

def create_tiny_subset(src_ascii, src_ann, dest_ascii, dest_ann, start_msg=1000, num_msgs=300):
    """
    Creates a tiny subset of the IRC dataset.
    Increased num_msgs to 300 to ensure we capture enough conversation context and links.
    """
    # Ensure directories exist
    os.makedirs(os.path.dirname(dest_ascii), exist_ok=True)
    os.makedirs(os.path.dirname(dest_ann), exist_ok=True)

    # Read ascii
    with open(src_ascii, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()
        # Ensure we don't go out of bounds
        end_msg = min(start_msg + num_msgs, len(all_lines))
        lines = all_lines[start_msg:end_msg]
    
    with open(dest_ascii, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    # Read annotations and filter
    filtered_ann = []
    if os.path.exists(src_ann):
        with open(src_ann, 'r', encoding='utf-8') as f:
            ann_lines = f.readlines()
        
        for line in ann_lines:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    # Format can be "child parent -" or "parent child -" depending on file
                    # But the logic remains: both indices must be in our window
                    idx1 = int(parts[0])
                    idx2 = int(parts[1])
                    
                    if start_msg <= idx1 < end_msg and start_msg <= idx2 < end_msg:
                        # Shift indices to be relative to the start of our tiny file
                        new_idx1 = idx1 - start_msg
                        new_idx2 = idx2 - start_msg
                        filtered_ann.append(f"{new_idx1} {new_idx2} -\n")
                except ValueError:
                    continue

    with open(dest_ann, 'w', encoding='utf-8') as f:
        f.writelines(filtered_ann)
    
    print(f"Created {dest_ascii} with {len(lines)} messages and {len(filtered_ann)} links.")

# Create tiny train
# Using start_msg=1000 because we saw many links in that range in the annotation file
create_tiny_subset(
    'data/train/2005-06-12.train-c.ascii.txt',
    'data/train/2005-06-12.train-c.annotation.txt',
    'data/tiny/train/tiny.train.ascii.txt',
    'data/tiny/train/tiny.train.annotation.txt',
    start_msg=1000, num_msgs=300
)

# Create tiny dev
create_tiny_subset(
    'data/dev/2004-11-15_03.ascii.txt',
    'data/dev/2004-11-15_03.annotation.txt',
    'data/tiny/dev/tiny.dev.ascii.txt',
    'data/tiny/dev/tiny.dev.annotation.txt',
    start_msg=1000, num_msgs=300
)

print("\nTiny dataset recreated in data/tiny/ with guaranteed gold links for testing.")
