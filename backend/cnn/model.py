import torch
import torch.nn as nn
from torchvision import models


DISASTER_CLASSES = ["fire", "flood", "earthquake", "hurricane", "normal"]


class DisasterCNN(nn.Module):
    """
    Fine-tuned ResNet-18 for disaster image classification.
    Uses pretrained ImageNet weights with a replaced final layer.
    """

    def __init__(self, num_classes: int = len(DISASTER_CLASSES), pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        base = models.resnet18(weights=weights)
        in_features = base.fc.in_features
        base.fc = nn.Linear(in_features, num_classes)
        self.network = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
