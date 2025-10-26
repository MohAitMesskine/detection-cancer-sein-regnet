"""
Training script for RegNetY breast cancer detection model.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path

from model import get_model
from dataset import create_data_loaders
from utils import save_checkpoint, load_checkpoint, plot_training_history


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """
    Train the model for one epoch.
    
    Args:
        model: The neural network model.
        dataloader: Training data loader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to train on.
        epoch: Current epoch number.
        
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device, epoch):
    """
    Validate the model.
    
    Args:
        model: The neural network model.
        dataloader: Validation data loader.
        criterion: Loss function.
        device: Device to validate on.
        epoch: Current epoch number.
        
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Val]')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
    
    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def train(args):
    """
    Main training function.
    
    Args:
        args: Command-line arguments.
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Create data loaders
    print("Creating data loaders...")
    data_loaders = create_data_loaders(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        augment=args.augment
    )
    
    print(f"Training samples: {data_loaders['train_size']}")
    print(f"Validation samples: {data_loaders['val_size']}")
    
    # Create model
    print(f"Creating {args.model_name} model...")
    model = get_model(
        model_name=args.model_name,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        dropout_rate=args.dropout_rate,
        device=device
    )
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, 
                          weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                 patience=5, verbose=True)
    
    # Load checkpoint if resuming
    start_epoch = 0
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, optimizer)
        if checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            best_val_acc = checkpoint.get('best_val_acc', 0.0)
            history = checkpoint.get('history', history)
            print(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, data_loaders['train'], criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, data_loaders['val'], criterion, device, epoch
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save checkpoint
        is_best = val_acc > best_val_acc
        best_val_acc = max(val_acc, best_val_acc)
        
        checkpoint_path = os.path.join(args.checkpoint_dir, 'last_checkpoint.pth')
        save_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            epoch,
            val_acc,
            best_val_acc,
            history
        )
        
        if is_best:
            best_checkpoint_path = os.path.join(args.checkpoint_dir, 'best_checkpoint.pth')
            save_checkpoint(
                best_checkpoint_path,
                model,
                optimizer,
                epoch,
                val_acc,
                best_val_acc,
                history
            )
            print(f"  New best model saved with validation accuracy: {val_acc:.2f}%")
    
    print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.2f}%")
    
    # Save training history
    history_path = os.path.join(args.output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    
    # Plot training history
    plot_path = os.path.join(args.output_dir, 'training_history.png')
    plot_training_history(history, plot_path)
    print(f"Training history saved to {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train RegNetY for breast cancer detection')
    
    # Data parameters
    parser.add_argument('--train_dir', type=str, default='data/train',
                       help='Path to training data directory')
    parser.add_argument('--val_dir', type=str, default='data/val',
                       help='Path to validation data directory')
    parser.add_argument('--image_size', type=int, default=224,
                       help='Input image size')
    parser.add_argument('--augment', action='store_true', default=True,
                       help='Apply data augmentation')
    
    # Model parameters
    parser.add_argument('--model_name', type=str, default='regnet_y_400mf',
                       choices=['regnet_y_400mf', 'regnet_y_800mf', 'regnet_y_1_6gf',
                               'regnet_y_3_2gf', 'regnet_y_8gf', 'regnet_y_16gf', 'regnet_y_32gf'],
                       help='RegNetY model variant')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained weights')
    parser.add_argument('--num_classes', type=int, default=2,
                       help='Number of classes')
    parser.add_argument('--dropout_rate', type=float, default=0.5,
                       help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--checkpoint_dir', type=str, default='models',
                       help='Checkpoint directory')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    train(args)
