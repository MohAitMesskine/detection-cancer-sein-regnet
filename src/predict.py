"""
Inference script for making predictions on new breast cancer images.
"""

import argparse
import torch
from torchvision import transforms
from PIL import Image
import os

from model import get_model
from utils import load_checkpoint


def load_image(image_path, image_size=224):
    """
    Load and preprocess an image for inference.
    
    Args:
        image_path (str): Path to the image.
        image_size (int): Size to resize the image to.
        
    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    return image_tensor


def predict(args):
    """
    Make predictions on a single image or directory of images.
    
    Args:
        args: Command-line arguments.
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    print(f"Loading {args.model_name} model...")
    model = get_model(
        model_name=args.model_name,
        pretrained=False,
        num_classes=args.num_classes,
        dropout_rate=args.dropout_rate,
        device=device
    )
    
    # Load checkpoint
    checkpoint = load_checkpoint(args.checkpoint, model)
    if checkpoint is None:
        print("Failed to load checkpoint. Exiting.")
        return
    
    model.eval()
    
    # Class names
    class_names = ['Benign', 'Malignant']
    
    # Process single image or directory
    if os.path.isfile(args.input):
        image_paths = [args.input]
    elif os.path.isdir(args.input):
        image_paths = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            image_paths.extend(Path(args.input).glob(ext))
        image_paths = [str(p) for p in image_paths]
    else:
        print(f"Invalid input path: {args.input}")
        return
    
    print(f"\nProcessing {len(image_paths)} image(s)...")
    
    # Make predictions
    with torch.no_grad():
        for img_path in image_paths:
            try:
                # Load and preprocess image
                image_tensor = load_image(img_path, args.image_size)
                image_tensor = image_tensor.to(device)
                
                # Forward pass
                outputs = model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                
                # Get prediction
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0, predicted_class].item()
                
                # Print results
                print(f"\nImage: {os.path.basename(img_path)}")
                print(f"  Prediction: {class_names[predicted_class]}")
                print(f"  Confidence: {confidence:.2%}")
                print(f"  Probabilities:")
                for i, class_name in enumerate(class_names):
                    print(f"    {class_name}: {probabilities[0, i].item():.2%}")
                
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Make predictions with RegNetY breast cancer model')
    
    # Input
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input image or directory of images')
    
    # Model parameters
    parser.add_argument('--model_name', type=str, default='regnet_y_400mf',
                       choices=['regnet_y_400mf', 'regnet_y_800mf', 'regnet_y_1_6gf',
                               'regnet_y_3_2gf', 'regnet_y_8gf', 'regnet_y_16gf', 'regnet_y_32gf'],
                       help='RegNetY model variant')
    parser.add_argument('--num_classes', type=int, default=2,
                       help='Number of classes')
    parser.add_argument('--dropout_rate', type=float, default=0.5,
                       help='Dropout rate')
    parser.add_argument('--image_size', type=int, default=224,
                       help='Input image size')
    
    # Checkpoint
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    
    args = parser.parse_args()
    
    from pathlib import Path
    predict(args)
