import os

def create_tiny_subset(src_ascii, src_ann, dest_ascii, dest_ann, start_msg=1000, num_msgs=100):
    # Read ascii
    with open(src_ascii, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()
        lines = all_lines[start_msg:start_msg+num_msgs]
    
    with open(dest_ascii, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    # Read annotations and filter
    # Note: The indices in the annotation file are absolute, 
    # but our data loader treats the file as a standalone conversation starting at 0.
    # So we need to shift the indices in the tiny annotation file.
    
    filtered_ann = []
    if os.path.exists(src_ann):
        with open(src_ann, 'r', encoding='utf-8') as f:
            ann_lines = f.readlines()
        
        for line in ann_lines:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    child = int(parts[0])
                    parent = int(parts[1])
                    # Check if both child and parent are within our window
                    if start_msg <= child < start_msg + num_msgs and start_msg <= parent < start_msg + num_msgs:
                        # Shift indices to be relative to the start of our tiny file
                        new_child = child - start_msg
                        new_parent = parent - start_msg
                        filtered_ann.append(f"{new_child} {new_parent} -\n")
                except ValueError:
                    continue

    with open(dest_ann, 'w', encoding='utf-8') as f:
        f.writelines(filtered_ann)

# Create tiny train (using a section of a train file)
# Let's check a train file first to see where links are, but usually they are throughout.
create_tiny_subset(
    'data/train/2005-06-12.train-c.ascii.txt',
    'data/train/2005-06-12.train-c.annotation.txt',
    'data/tiny/train/tiny.train.ascii.txt',
    'data/tiny/train/tiny.train.annotation.txt',
    start_msg=100, num_msgs=100
)

# Create tiny dev (using the known range in 2004-11-15_03)
create_tiny_subset(
    'data/dev/2004-11-15_03.ascii.txt',
    'data/dev/2004-11-15_03.annotation.txt',
    'data/tiny/dev/tiny.dev.ascii.txt',
    'data/tiny/dev/tiny.dev.annotation.txt',
    start_msg=1000, num_msgs=100
)

print("Tiny dataset recreated in data/tiny/ with shifted indices to ensure links are present.")
