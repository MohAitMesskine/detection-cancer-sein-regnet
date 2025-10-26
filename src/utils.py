"""
Utility functions for training, evaluation, and visualization.
"""

import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import json


def save_checkpoint(filepath, model, optimizer, epoch, val_acc, best_val_acc, history):
    """
    Save model checkpoint.
    
    Args:
        filepath (str): Path to save the checkpoint.
        model: Model to save.
        optimizer: Optimizer state to save.
        epoch (int): Current epoch.
        val_acc (float): Current validation accuracy.
        best_val_acc (float): Best validation accuracy so far.
        history (dict): Training history.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'best_val_acc': best_val_acc,
        'history': history
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(filepath, model, optimizer=None):
    """
    Load model checkpoint.
    
    Args:
        filepath (str): Path to the checkpoint file.
        model: Model to load the state into.
        optimizer: Optional optimizer to load the state into.
        
    Returns:
        dict: Checkpoint dictionary or None if file doesn't exist.
    """
    if not os.path.exists(filepath):
        print(f"Checkpoint file {filepath} not found.")
        return None
    
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Checkpoint loaded from {filepath}")
    return checkpoint


def plot_training_history(history, save_path=None):
    """
    Plot training and validation loss and accuracy.
    
    Args:
        history (dict): Dictionary with 'train_loss', 'train_acc', 'val_loss', 'val_acc'.
        save_path (str): Path to save the plot. If None, displays the plot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot accuracy
    axes[1].plot(history['train_acc'], label='Train Accuracy', marker='o')
    axes[1].plot(history['val_acc'], label='Val Accuracy', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        class_names: List of class names.
        save_path (str): Path to save the plot. If None, displays the plot.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_roc_curve(y_true, y_scores, save_path=None):
    """
    Plot ROC curve for binary classification.
    
    Args:
        y_true: True binary labels.
        y_scores: Predicted probabilities for the positive class.
        save_path (str): Path to save the plot. If None, displays the plot.
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    return roc_auc


def calculate_metrics(y_true, y_pred, y_scores=None):
    """
    Calculate various classification metrics.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_scores: Predicted probabilities (optional, for AUC calculation).
        
    Returns:
        dict: Dictionary with various metrics.
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='binary', zero_division=0)
    }
    
    if y_scores is not None:
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        metrics['auc'] = auc(fpr, tpr)
    
    return metrics


def save_metrics(metrics, filepath):
    """
    Save metrics to a JSON file.
    
    Args:
        metrics (dict): Dictionary of metrics.
        filepath (str): Path to save the metrics.
    """
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {filepath}")


def get_predictions(model, dataloader, device):
    """
    Get predictions from the model on a dataset.
    
    Args:
        model: Trained model.
        dataloader: Data loader.
        device: Device to run inference on.
        
    Returns:
        tuple: (y_true, y_pred, y_scores) - true labels, predicted labels, and prediction scores.
    """
    model.eval()
    y_true = []
    y_pred = []
    y_scores = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            
            # Get probabilities using softmax
            probabilities = torch.softmax(outputs, dim=1)
            
            # Get predictions
            _, predicted = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_scores.extend(probabilities[:, 1].cpu().numpy())  # Probability of positive class
    
    return np.array(y_true), np.array(y_pred), np.array(y_scores)


if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")
    
    # Create dummy history
    history = {
        'train_loss': [0.5, 0.4, 0.3, 0.2],
        'train_acc': [70, 75, 80, 85],
        'val_loss': [0.6, 0.5, 0.4, 0.3],
        'val_acc': [65, 70, 75, 80]
    }
    
    # Test plotting
    plot_training_history(history)
    print("Utility functions test passed!")
