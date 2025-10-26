"""
Evaluation script for the trained RegNetY model.
"""

import argparse
import torch
import os
from pathlib import Path

from model import get_model
from dataset import create_data_loaders
from utils import (load_checkpoint, get_predictions, calculate_metrics,
                  save_metrics, plot_confusion_matrix, plot_roc_curve)


def evaluate(args):
    """
    Evaluate the trained model on test/validation data.
    
    Args:
        args: Command-line arguments.
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    print(f"Creating {args.model_name} model...")
    model = get_model(
        model_name=args.model_name,
        pretrained=False,
        num_classes=args.num_classes,
        dropout_rate=args.dropout_rate,
        device=device
    )
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = load_checkpoint(args.checkpoint, model)
    if checkpoint is None:
        print("Failed to load checkpoint. Exiting.")
        return
    
    # Create data loader for evaluation
    print("Creating data loader...")
    data_loaders = create_data_loaders(
        train_dir=args.data_dir,  # Using same for both since we only need one
        val_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        augment=False  # No augmentation for evaluation
    )
    
    eval_loader = data_loaders['val']
    print(f"Evaluation samples: {data_loaders['val_size']}")
    
    # Get predictions
    print("Getting predictions...")
    y_true, y_pred, y_scores = get_predictions(model, eval_loader, device)
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(y_true, y_pred, y_scores)
    
    print("\nEvaluation Results:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1-Score: {metrics['f1_score']:.4f}")
    if 'auc' in metrics:
        print(f"  AUC-ROC: {metrics['auc']:.4f}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save metrics
    metrics_path = os.path.join(args.output_dir, 'evaluation_metrics.json')
    save_metrics(metrics, metrics_path)
    
    # Plot confusion matrix
    class_names = ['Benign', 'Malignant']
    cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(y_true, y_pred, class_names, cm_path)
    
    # Plot ROC curve
    roc_path = os.path.join(args.output_dir, 'roc_curve.png')
    plot_roc_curve(y_true, y_scores, roc_path)
    
    print(f"\nEvaluation complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate RegNetY breast cancer detection model')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to evaluation data directory')
    parser.add_argument('--image_size', type=int, default=224,
                       help='Input image size')
    
    # Model parameters
    parser.add_argument('--model_name', type=str, default='regnet_y_400mf',
                       choices=['regnet_y_400mf', 'regnet_y_800mf', 'regnet_y_1_6gf',
                               'regnet_y_3_2gf', 'regnet_y_8gf', 'regnet_y_16gf', 'regnet_y_32gf'],
                       help='RegNetY model variant')
    parser.add_argument('--num_classes', type=int, default=2,
                       help='Number of classes')
    parser.add_argument('--dropout_rate', type=float, default=0.5,
                       help='Dropout rate')
    
    # Checkpoint
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    
    # Evaluation parameters
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='results/evaluation',
                       help='Output directory for evaluation results')
    
    args = parser.parse_args()
    
    evaluate(args)
