from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainingConfig:
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 20
    num_classes: int = 10
    image_size: int = 32
    patch_size: int = 4
    embed_dim: int = 128
    transformer_depth: int = 4
    transformer_heads: int = 4
    validation_split: float = 0.1
    num_workers: int = 2
    data_dir: str = "./data"
    checkpoint_dir: str = "./checkpoints"
    seed: int = 42

    @property
    def checkpoint_path(self) -> Path:
        return Path(self.checkpoint_dir)
