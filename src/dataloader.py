from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms


@dataclass
class CIFAR10DataConfig:
    data_dir: str = "./data"
    batch_size: int = 64
    num_workers: int = 2
    validation_split: float = 0.1
    image_size: int = 32
    use_augmentation: bool = True
    normalize: bool = True
    seed: int = 42


class CIFAR10DataModule:
    """Builds CIFAR-10 datasets and dataloaders for training workflows."""

    classes = (
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    def __init__(self, config: Optional[CIFAR10DataConfig] = None) -> None:
        self.config = config or CIFAR10DataConfig()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _build_transforms(self) -> Tuple[transforms.Compose, transforms.Compose]:
        transform_steps = []
        eval_steps = []

        if self.config.image_size != 32:
            resize = transforms.Resize((self.config.image_size, self.config.image_size))
            transform_steps.append(resize)
            eval_steps.append(resize)

        if self.config.use_augmentation:
            transform_steps.extend(
                [
                    transforms.RandomCrop(self.config.image_size, padding=4),
                    transforms.RandomHorizontalFlip(),
                ]
            )

        transform_steps.append(transforms.ToTensor())
        eval_steps.append(transforms.ToTensor())

        if self.config.normalize:
            normalize = transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616),
            )
            transform_steps.append(normalize)
            eval_steps.append(normalize)

        return transforms.Compose(transform_steps), transforms.Compose(eval_steps)

    def prepare_data(self) -> None:
        """Downloads CIFAR-10 if it is not already available locally."""
        datasets.CIFAR10(root=self.config.data_dir, train=True, download=True)
        datasets.CIFAR10(root=self.config.data_dir, train=False, download=True)

    def setup(self) -> None:
        """Creates train, validation, and test dataset splits."""
        train_transform, eval_transform = self._build_transforms()

        full_train_dataset = datasets.CIFAR10(
            root=self.config.data_dir,
            train=True,
            download=False,
            transform=train_transform,
        )

        val_source_dataset = datasets.CIFAR10(
            root=self.config.data_dir,
            train=True,
            download=False,
            transform=eval_transform,
        )

        test_dataset = datasets.CIFAR10(
            root=self.config.data_dir,
            train=False,
            download=False,
            transform=eval_transform,
        )

        val_size = int(len(full_train_dataset) * self.config.validation_split)
        train_size = len(full_train_dataset) - val_size

        generator = torch.Generator().manual_seed(self.config.seed)
        train_subset, val_subset_with_indices = random_split(
            full_train_dataset, [train_size, val_size], generator=generator
        )

        val_subset = Subset(val_source_dataset, val_subset_with_indices.indices)

        self.train_dataset = train_subset
        self.val_dataset = val_subset
        self.test_dataset = test_dataset

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Call setup() before requesting dataloaders.")

        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("Call setup() before requesting dataloaders.")

        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("Call setup() before requesting dataloaders.")

        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

    def get_class_names(self) -> Tuple[str, ...]:
        return self.classes
