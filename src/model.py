"""
RegNetY Model for Breast Cancer Detection
This module implements the RegNetY architecture for binary classification
of breast cancer images.
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import RegNet_Y_400MF_Weights, RegNet_Y_800MF_Weights, RegNet_Y_1_6GF_Weights, RegNet_Y_3_2GF_Weights, RegNet_Y_8GF_Weights, RegNet_Y_16GF_Weights, RegNet_Y_32GF_Weights


class RegNetYBreastCancer(nn.Module):
    """
    RegNetY-based model for breast cancer detection.
    
    This model uses a pre-trained RegNetY backbone from torchvision
    and adapts it for binary classification (malignant vs benign).
    
    Args:
        model_name (str): Name of the RegNetY variant to use.
                         Options: 'regnet_y_400mf', 'regnet_y_800mf', 
                                 'regnet_y_1_6gf', 'regnet_y_3_2gf', etc.
        pretrained (bool): Whether to use pre-trained weights from ImageNet.
        num_classes (int): Number of output classes (default: 2 for binary classification).
        dropout_rate (float): Dropout rate for regularization (default: 0.5).
    """
    
    def __init__(self, model_name='regnet_y_400mf', pretrained=True, 
                 num_classes=2, dropout_rate=0.5):
        super(RegNetYBreastCancer, self).__init__()
        
        # Map model names to weights
        weights_map = {
            'regnet_y_400mf': RegNet_Y_400MF_Weights.IMAGENET1K_V2 if pretrained else None,
            'regnet_y_800mf': RegNet_Y_800MF_Weights.IMAGENET1K_V2 if pretrained else None,
            'regnet_y_1_6gf': RegNet_Y_1_6GF_Weights.IMAGENET1K_V2 if pretrained else None,
            'regnet_y_3_2gf': RegNet_Y_3_2GF_Weights.IMAGENET1K_V2 if pretrained else None,
            'regnet_y_8gf': RegNet_Y_8GF_Weights.IMAGENET1K_V2 if pretrained else None,
            'regnet_y_16gf': RegNet_Y_16GF_Weights.IMAGENET1K_V2 if pretrained else None,
            'regnet_y_32gf': RegNet_Y_32GF_Weights.IMAGENET1K_V2 if pretrained else None,
        }
        
        # Load the RegNetY model
        if model_name == 'regnet_y_400mf':
            self.backbone = models.regnet_y_400mf(weights=weights_map[model_name])
        elif model_name == 'regnet_y_800mf':
            self.backbone = models.regnet_y_800mf(weights=weights_map[model_name])
        elif model_name == 'regnet_y_1_6gf':
            self.backbone = models.regnet_y_1_6gf(weights=weights_map[model_name])
        elif model_name == 'regnet_y_3_2gf':
            self.backbone = models.regnet_y_3_2gf(weights=weights_map[model_name])
        elif model_name == 'regnet_y_8gf':
            self.backbone = models.regnet_y_8gf(weights=weights_map[model_name])
        elif model_name == 'regnet_y_16gf':
            self.backbone = models.regnet_y_16gf(weights=weights_map[model_name])
        elif model_name == 'regnet_y_32gf':
            self.backbone = models.regnet_y_32gf(weights=weights_map[model_name])
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        
        # Get the number of features from the original classifier
        in_features = self.backbone.fc.in_features
        
        # Replace the classifier with a custom one for breast cancer detection
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        return self.backbone(x)


def get_model(model_name='regnet_y_400mf', pretrained=True, num_classes=2, 
              dropout_rate=0.5, device='cuda'):
    """
    Factory function to create and initialize a RegNetY model.
    
    Args:
        model_name (str): Name of the RegNetY variant.
        pretrained (bool): Whether to use pre-trained weights.
        num_classes (int): Number of output classes.
        dropout_rate (float): Dropout rate for regularization.
        device (str): Device to move the model to ('cuda' or 'cpu').
        
    Returns:
        RegNetYBreastCancer: Initialized model ready for training/inference.
    """
    model = RegNetYBreastCancer(
        model_name=model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        dropout_rate=dropout_rate
    )
    
    # Move model to device
    if device == 'cuda' and torch.cuda.is_available():
        model = model.cuda()
    else:
        model = model.cpu()
    
    return model


if __name__ == "__main__":
    # Test the model
    print("Testing RegNetY model...")
    model = get_model(model_name='regnet_y_400mf', pretrained=False, device='cpu')
    
    # Create a random input tensor
    x = torch.randn(2, 3, 224, 224)
    
    # Forward pass
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("Model test passed!")
