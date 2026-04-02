"""Model package for CNN and transformer architectures."""

from .cnn import CIFAR10CNN
from .vit import ViTLite

__all__ = ["CIFAR10CNN", "ViTLite"]

