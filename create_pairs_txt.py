#!/usr/bin/env python3
import os
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="Create paired list file from low and high image directories.")
    parser.add_argument(
        "--base_dir", 
        default = "/scratch/user/u.ok285885/data/LOL-v1/our485/",
        help="Base directory containing 'low' and 'high' subdirectories"
    )
    parser.add_argument(
        "--output", 
        default="/scratch/user/u.ok285885/data/LOL-v1/LOLv1_val.txt", 
        help="Output text file name (default: paired_list.txt)"
    )
    args = parser.parse_args()

    base_dir = args.base_dir
    low_dir = os.path.join(base_dir, "low")
    high_dir = os.path.join(base_dir, "high")

    # Check that both subdirectories exist
    if not os.path.isdir(low_dir):
        sys.exit(f"Error: '{low_dir}' directory does not exist.")
    if not os.path.isdir(high_dir):
        sys.exit(f"Error: '{high_dir}' directory does not exist.")

    # List files in both directories (only files, ignore hidden files)
    low_files = sorted([f for f in os.listdir(low_dir) if os.path.isfile(os.path.join(low_dir, f)) and not f.startswith(".")])
    high_files = sorted([f for f in os.listdir(high_dir) if os.path.isfile(os.path.join(high_dir, f)) and not f.startswith(".")])

    # Create sets of filenames (you can further process names if needed e.g., strip extensions)
    set_low = set(low_files)
    set_high = set(high_files)

    common_files = sorted(list(set_low.intersection(set_high)))
    missing_in_high = sorted(list(set_low - set_high))
    missing_in_low = sorted(list(set_high - set_low))

    # Warn if there are non-pairs
    if missing_in_high:
        print("WARNING: The following files are present in 'low' but missing in 'high':")
        for f in missing_in_high:
            print("  ", f)
    if missing_in_low:
        print("WARNING: The following files are present in 'high' but missing in 'low':")
        for f in missing_in_low:
            print("  ", f)

    total_pairs = len(common_files)
    print(f"\nTotal pairs found: {total_pairs}")
    print(f"Files only in low: {len(missing_in_high)}")
    print(f"Files only in high: {len(missing_in_low)}")

    # Write the pairs to output file: each line is "low_path high_path"
    output_path = os.path.abspath(args.output)
    with open(output_path, "w") as out_f:
        for f in common_files:
            low_path = os.path.abspath(os.path.join(low_dir, f))
            high_path = os.path.abspath(os.path.join(high_dir, f))
            out_f.write(f"{low_path} {high_path}\n")

    print(f"\nPaired list written to: {output_path}")

if __name__ == "__main__":
    main()
