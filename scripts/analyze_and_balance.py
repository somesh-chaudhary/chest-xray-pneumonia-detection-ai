"""
Dataset Analysis Script for Chest X-Ray Pneumonia Detection
Analyzes class distribution and identifies potential bias in the raw dataset
"""

import os
import shutil
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split

def find_dataset_path():
    """Find the correct dataset path from possible locations"""
    possible_paths = [
        Path("../data/raw/chest_xray"),        # From scripts folder
        Path("data/raw/chest_xray"),           # From root folder
        Path("../data/raw/archive/chest_xrays/chest_xrays"),  # Nested structure
        Path("../data/raw/chest_xrays"),       # Alternative name
    ]

    for path in possible_paths:
        if path.exists():
            print(f"Found dataset at: {path}")
            return path
    
    print("Dataset not found! Checked these locations:")
    for path in possible_paths:
        print(f" - {path.absolute()}")
    return None

def analyze_current_data():
    """Analyze the current dataset distribution and identify bias"""
    print("Analyzing dataset distribution...")
    print("=" * 50)
    
    # Find dataset path
    base_path = find_dataset_path()
    if base_path is None:
        print("\nManual check required:")
        print("1. Open your file explorer")
        print("2. Navigate to chest_xray_project/data/raw/")
        print("3. Verify the folder structure")
        return False, None, False

    # Check for required folders
    required_folders = ['train', 'test', 'val']
    missing_folders = []
    for folder in required_folders:
        if not (base_path / folder).exists():
            missing_folders.append(folder)

    if missing_folders:
        print(f"Missing folders: {missing_folders}")
        return False, None, False

    # Analyze data distribution
    splits = ['train', 'test', 'val']
    classes = ['normal', 'pneumonia', 'NORMAL', 'PNEUMONIA']  # Try both cases
    data_info = []
    total_images = 0

    for split in splits:
        split_path = base_path / split
        print(f"\n{split.upper()} SET:")
        split_total = 0
        
        # Find the correct class folder names
        found_classes = []
        for cls in classes:
            cls_path = split_path / cls
            if cls_path.exists():
                found_classes.append(cls)

        if not found_classes:
            print(f"  No class folders found in {split}")
            continue

        for cls in found_classes:
            cls_path = split_path / cls
            # Count image files
            image_files = (list(cls_path.glob("*.jpeg")) + 
                          list(cls_path.glob("*.jpg")) + 
                          list(cls_path.glob("*.png")))
            count = len(image_files)
            print(f"  {cls.upper()}: {count:,} images")
            
            data_info.append({
                'Split': split,
                'Class': cls.upper(),
                'Count': count
            })
            split_total += count

        print(f"  Total {split}: {split_total:,} images")
        total_images += split_total

    if total_images == 0:
        print("No images found!")
        return False, None, False

    print(f"\nTOTAL DATASET: {total_images:,} images")

    # Create analysis DataFrame
    df = pd.DataFrame(data_info)
    class_totals = df.groupby('Class')['Count'].sum()
    
    print(f"\nCLASS DISTRIBUTION:")
    print("=" * 30)
    for cls, count in class_totals.items():
        percentage = (count / total_images) * 100
        print(f"{cls}: {count:,} images ({percentage:.1f}%)")

    # Check for class imbalance
    class_counts = list(class_totals.values)
    if len(class_counts) >= 2:
        imbalance_ratio = max(class_counts) / min(class_counts)
        print(f"\nImbalance Ratio: {imbalance_ratio:.2f}:1")
        
        bias_risk = imbalance_ratio > 2.0
        if bias_risk:
            print("WARNING: Significant class imbalance detected!")
            print("This may cause model bias and poor performance.")
        else:
            print("Classes are reasonably balanced")
    else:
        bias_risk = False

    return True, df, bias_risk

def main():
    """Main analysis function"""
    print("Starting dataset analysis...")
    
    try:
        success, data_df, has_bias = analyze_current_data()
        
        if success:
            print(f"\nAnalysis complete!")
            if has_bias:
                print("Dataset requires balancing to prevent model bias")
            else:
                print("Dataset is ready for training")
        else:
            print("\nAnalysis failed - check your dataset structure")
            
    except Exception as e:
        print(f"Error occurred: {e}")
        print("\nDebug information:")
        print(f"Current working directory: {os.getcwd()}")
        print("Contents of current directory:")
        for item in Path(".").iterdir():
            print(f"  - {item.name}")

if __name__ == "__main__":
    main()
