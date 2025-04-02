#!/usr/bin/env python3
import os
import sys
import argparse

def process_camera_folder(camera_dir):
    """Process a camera folder (e.g., Huawei or Nikon) that should contain 'low' and 'high' subdirectories.
    Returns a tuple (pairs, missing_low, missing_high) where:
      - pairs is a list of strings ("low_path high_path")
      - missing_low is list of filenames in low but not in high.
      - missing_high is list of filenames in high but not in low.
    """
    low_dir = os.path.join(camera_dir, "low")
    high_dir = os.path.join(camera_dir, "high")
    if not os.path.isdir(low_dir):
        print(f"WARNING: '{low_dir}' does not exist. Skipping {camera_dir}.")
        return [], [], []
    if not os.path.isdir(high_dir):
        print(f"WARNING: '{high_dir}' does not exist. Skipping {camera_dir}.")
        return [], [], []
    
    low_files = sorted([f for f in os.listdir(low_dir) if os.path.isfile(os.path.join(low_dir, f)) and not f.startswith(".")])
    high_files = sorted([f for f in os.listdir(high_dir) if os.path.isfile(os.path.join(high_dir, f)) and not f.startswith(".")])
    
    set_low = set(low_files)
    set_high = set(high_files)
    common_files = sorted(list(set_low.intersection(set_high)))
    missing_in_high = sorted(list(set_low - set_high))
    missing_in_low = sorted(list(set_high - set_low))
    
    if missing_in_high:
        print(f"WARNING in {camera_dir}: The following files are in 'low' but missing in 'high': {missing_in_high}")
    if missing_in_low:
        print(f"WARNING in {camera_dir}: The following files are in 'high' but missing in 'low': {missing_in_low}")
    
    pairs = []
    for f in common_files:
        low_path = os.path.abspath(os.path.join(low_dir, f))
        high_path = os.path.abspath(os.path.join(high_dir, f))
        pairs.append(f"{low_path} {high_path}")
    
    return pairs, missing_in_high, missing_in_low

def main():
    parser = argparse.ArgumentParser(description="Generate paired list file for LSRW dataset with multiple camera subfolders.")
    parser.add_argument("--base_dir", default="/scratch/user/u.ok285885/data/LSRW", 
                        help="Base directory containing camera subdirectories (e.g., Huawei, Nikon)")
    parser.add_argument("--output", default="/scratch/user/u.ok285885/data/LSRW_pairs.txt", 
                        help="Output text file name (default: LSRW_pairs.txt)")
    args = parser.parse_args()
    
    base_dir = args.base_dir
    if not os.path.isdir(base_dir):
        sys.exit(f"Error: Base directory '{base_dir}' does not exist.")
    
    total_pairs = 0
    overall_pairs = []
    
    # Iterate through each subdirectory in base_dir (each representing a camera source)
    for subfolder in sorted(os.listdir(base_dir)):
        camera_dir = os.path.join(base_dir, subfolder)
        if os.path.isdir(camera_dir):
            print(f"Processing camera folder: {camera_dir}")
            pairs, missing_low, missing_high = process_camera_folder(camera_dir)
            num_pairs = len(pairs)
            print(f"Found {num_pairs} paired images in {subfolder}.")
            overall_pairs.extend(pairs)
            total_pairs += num_pairs
        else:
            print(f"Skipping non-directory entry: {subfolder}")
    
    print(f"\nTotal pairs found across all camera folders: {total_pairs}")
    
    output_path = os.path.abspath(args.output)
    with open(output_path, "w") as out_f:
        for line in overall_pairs:
            out_f.write(line + "\n")
    
    print(f"\nPaired list written to: {output_path}")

if __name__ == "__main__":
    main()
