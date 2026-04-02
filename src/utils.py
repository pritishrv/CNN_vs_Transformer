from pathlib import Path

import torch
from torch import nn

from src.config import TrainingConfig
from src.models import CIFAR10CNN, ViTLite


def build_model(model_name: str, config: TrainingConfig) -> nn.Module:
    if model_name == "cnn":
        return CIFAR10CNN(num_classes=config.num_classes)
    if model_name == "vit":
        return ViTLite(
            image_size=config.image_size,
            patch_size=config.patch_size,
            num_classes=config.num_classes,
            embed_dim=config.embed_dim,
            depth=config.transformer_depth,
            num_heads=config.transformer_heads,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def load_checkpoint_config(checkpoint: dict) -> TrainingConfig:
    config = TrainingConfig()
    for key, value in checkpoint.get("config", {}).items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config


def load_model_from_checkpoint(
    model_name: str,
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[nn.Module, TrainingConfig, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = load_checkpoint_config(checkpoint)
    model = build_model(model_name, config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, config, checkpoint
