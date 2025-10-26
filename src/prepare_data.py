"""
Data preparation script for breast cancer dataset.
This script helps organize and split the dataset for training.
"""

import os
import shutil
from pathlib import Path
import random
from tqdm import tqdm
import argparse


def split_dataset(source_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        source_dir (str): Source directory containing class folders.
        output_dir (str): Output directory for split datasets.
        train_ratio (float): Ratio of training data.
        val_ratio (float): Ratio of validation data.
        test_ratio (float): Ratio of test data.
        seed (int): Random seed for reproducibility.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    random.seed(seed)
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Get all class folders
    class_folders = [f for f in source_path.iterdir() if f.is_dir()]
    
    if not class_folders:
        print(f"No class folders found in {source_dir}")
        return
    
    print(f"Found {len(class_folders)} classes:")
    for folder in class_folders:
        print(f"  - {folder.name}")
    
    # Create output structure
    for split in ['train', 'val', 'test']:
        for class_folder in class_folders:
            split_dir = output_path / split / class_folder.name
            split_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each class
    for class_folder in class_folders:
        print(f"\nProcessing class: {class_folder.name}")
        
        # Get all images
        images = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
            images.extend(list(class_folder.glob(ext)))
        
        print(f"  Found {len(images)} images")
        
        # Shuffle images
        random.shuffle(images)
        
        # Calculate split indices
        n_train = int(len(images) * train_ratio)
        n_val = int(len(images) * val_ratio)
        
        train_images = images[:n_train]
        val_images = images[n_train:n_train + n_val]
        test_images = images[n_train + n_val:]
        
        print(f"  Split: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")
        
        # Copy files
        splits = {
            'train': train_images,
            'val': val_images,
            'test': test_images
        }
        
        for split_name, split_images in splits.items():
            dest_dir = output_path / split_name / class_folder.name
            for img_path in tqdm(split_images, desc=f"  Copying to {split_name}"):
                dest_path = dest_dir / img_path.name
                shutil.copy2(img_path, dest_path)
    
    print("\nDataset split completed!")
    print(f"Output directory: {output_dir}")
    
    # Print summary
    print("\nSummary:")
    for split in ['train', 'val', 'test']:
        split_path = output_path / split
        total = 0
        for class_folder in class_folders:
            class_path = split_path / class_folder.name
            n_images = len(list(class_path.glob('*.*')))
            print(f"  {split}/{class_folder.name}: {n_images} images")
            total += n_images
        print(f"  {split} total: {total} images")


def organize_flat_dataset(source_dir, output_dir, class_labels_file=None):
    """
    Organize a flat dataset (all images in one folder) into class folders.
    
    Args:
        source_dir (str): Source directory with all images.
        output_dir (str): Output directory for organized dataset.
        class_labels_file (str): CSV file with image names and labels.
    """
    import pandas as pd
    
    if class_labels_file is None:
        print("Error: class_labels_file is required for flat dataset organization")
        return
    
    # Read labels
    df = pd.read_csv(class_labels_file)
    print(f"Loaded {len(df)} labels from {class_labels_file}")
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Create class folders
    classes = df['label'].unique()
    for cls in classes:
        (output_path / str(cls)).mkdir(parents=True, exist_ok=True)
    
    # Copy files to appropriate folders
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Organizing files"):
        img_name = row['image_name']
        label = row['label']
        
        src_file = source_path / img_name
        dest_file = output_path / str(label) / img_name
        
        if src_file.exists():
            shutil.copy2(src_file, dest_file)
        else:
            print(f"Warning: {src_file} not found")
    
    print(f"\nDataset organized into {output_dir}")
    for cls in classes:
        n_images = len(list((output_path / str(cls)).glob('*.*')))
        print(f"  Class {cls}: {n_images} images")


def verify_dataset(data_dir):
    """
    Verify dataset structure and count images.
    
    Args:
        data_dir (str): Path to dataset directory.
    """
    data_path = Path(data_dir)
    
    print(f"Verifying dataset in: {data_dir}")
    print(f"{'='*60}")
    
    if not data_path.exists():
        print(f"Error: Directory {data_dir} does not exist")
        return
    
    # Check for split folders
    splits = ['train', 'val', 'test']
    for split in splits:
        split_path = data_path / split
        if split_path.exists():
            print(f"\n{split.upper()} SET:")
            class_folders = [f for f in split_path.iterdir() if f.is_dir()]
            for class_folder in class_folders:
                n_images = len(list(class_folder.glob('*.*')))
                print(f"  {class_folder.name}: {n_images} images")
        else:
            print(f"\n{split.upper()} SET: Not found")
    
    print(f"\n{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare breast cancer dataset')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Split command
    split_parser = subparsers.add_parser('split', help='Split dataset into train/val/test')
    split_parser.add_argument('--source_dir', type=str, required=True,
                             help='Source directory with class folders')
    split_parser.add_argument('--output_dir', type=str, required=True,
                             help='Output directory for split datasets')
    split_parser.add_argument('--train_ratio', type=float, default=0.7,
                             help='Training data ratio (default: 0.7)')
    split_parser.add_argument('--val_ratio', type=float, default=0.15,
                             help='Validation data ratio (default: 0.15)')
    split_parser.add_argument('--test_ratio', type=float, default=0.15,
                             help='Test data ratio (default: 0.15)')
    split_parser.add_argument('--seed', type=int, default=42,
                             help='Random seed (default: 42)')
    
    # Organize command
    organize_parser = subparsers.add_parser('organize', help='Organize flat dataset into class folders')
    organize_parser.add_argument('--source_dir', type=str, required=True,
                                help='Source directory with all images')
    organize_parser.add_argument('--output_dir', type=str, required=True,
                                help='Output directory for organized dataset')
    organize_parser.add_argument('--labels_file', type=str, required=True,
                                help='CSV file with image names and labels')
    
    # Verify command
    verify_parser = subparsers.add_parser('verify', help='Verify dataset structure')
    verify_parser.add_argument('--data_dir', type=str, required=True,
                              help='Dataset directory to verify')
    
    args = parser.parse_args()
    
    if args.command == 'split':
        split_dataset(
            args.source_dir,
            args.output_dir,
            args.train_ratio,
            args.val_ratio,
            args.test_ratio,
            args.seed
        )
    elif args.command == 'organize':
        organize_flat_dataset(
            args.source_dir,
            args.output_dir,
            args.labels_file
        )
    elif args.command == 'verify':
        verify_dataset(args.data_dir)
    else:
        parser.print_help()
