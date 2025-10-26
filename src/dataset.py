"""
Data loading and preprocessing utilities for breast cancer detection.
Supports loading images from the Kaggle breast cancer dataset.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from pathlib import Path


class BreastCancerDataset(Dataset):
    """
    Custom Dataset for breast cancer images.
    
    The dataset is expected to have the following structure:
    - data/raw/
        - benign/
            - image1.png
            - image2.png
            ...
        - malignant/
            - image1.png
            - image2.png
            ...
    
    Or alternatively, a CSV file with image paths and labels.
    """
    
    def __init__(self, data_dir=None, csv_file=None, transform=None, mode='folder'):
        """
        Args:
            data_dir (str): Path to the data directory containing class folders.
            csv_file (str): Path to CSV file with 'image_path' and 'label' columns.
            transform (callable, optional): Optional transform to be applied on images.
            mode (str): Either 'folder' or 'csv' to specify data loading method.
        """
        self.transform = transform
        self.mode = mode
        self.images = []
        self.labels = []
        
        if mode == 'folder' and data_dir:
            self._load_from_folders(data_dir)
        elif mode == 'csv' and csv_file:
            self._load_from_csv(csv_file)
        else:
            raise ValueError("Either data_dir (for folder mode) or csv_file (for csv mode) must be provided")
    
    def _load_from_folders(self, data_dir):
        """Load images from folder structure with class subdirectories."""
        data_path = Path(data_dir)
        
        # Class mapping: benign=0, malignant=1
        class_to_idx = {'benign': 0, 'malignant': 1}
        
        for class_name, label in class_to_idx.items():
            class_dir = data_path / class_name
            if class_dir.exists():
                for img_file in class_dir.glob('*.png'):
                    self.images.append(str(img_file))
                    self.labels.append(label)
                for img_file in class_dir.glob('*.jpg'):
                    self.images.append(str(img_file))
                    self.labels.append(label)
                for img_file in class_dir.glob('*.jpeg'):
                    self.images.append(str(img_file))
                    self.labels.append(label)
        
        print(f"Loaded {len(self.images)} images from {data_dir}")
    
    def _load_from_csv(self, csv_file):
        """Load images from CSV file with image paths and labels."""
        df = pd.read_csv(csv_file)
        self.images = df['image_path'].tolist()
        self.labels = df['label'].tolist()
        print(f"Loaded {len(self.images)} images from {csv_file}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample.
            
        Returns:
            tuple: (image, label) where image is a transformed PIL Image and label is an integer.
        """
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_data_transforms(image_size=224, augment=True):
    """
    Get data transformation pipelines for training and validation.
    
    Args:
        image_size (int): Target size for images (default: 224).
        augment (bool): Whether to apply data augmentation for training.
        
    Returns:
        dict: Dictionary with 'train' and 'val' transformations.
    """
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return {'train': train_transform, 'val': val_transform}


def create_data_loaders(train_dir, val_dir, batch_size=32, num_workers=4, 
                       image_size=224, augment=True):
    """
    Create DataLoaders for training and validation.
    
    Args:
        train_dir (str): Path to training data directory.
        val_dir (str): Path to validation data directory.
        batch_size (int): Batch size for data loaders.
        num_workers (int): Number of worker processes for data loading.
        image_size (int): Target size for images.
        augment (bool): Whether to apply data augmentation for training.
        
    Returns:
        dict: Dictionary with 'train' and 'val' DataLoaders.
    """
    transforms_dict = get_data_transforms(image_size, augment)
    
    # Create datasets
    train_dataset = BreastCancerDataset(
        data_dir=train_dir,
        transform=transforms_dict['train'],
        mode='folder'
    )
    
    val_dataset = BreastCancerDataset(
        data_dir=val_dir,
        transform=transforms_dict['val'],
        mode='folder'
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'train_size': len(train_dataset),
        'val_size': len(val_dataset)
    }


if __name__ == "__main__":
    # Test the data loading
    print("Testing data loading utilities...")
    transforms_dict = get_data_transforms(image_size=224, augment=True)
    print("Data transforms created successfully!")
    print(f"Train transform: {transforms_dict['train']}")
    print(f"Val transform: {transforms_dict['val']}")
