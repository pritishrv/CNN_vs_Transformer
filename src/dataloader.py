from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms


@dataclass
class CIFAR10DataConfig:
    data_dir: str = "./cifar10"
    manifest_path: Optional[str] = None
    batch_size: int = 64
    num_workers: int = 2
    validation_split: float = 0.1
    image_size: int = 32
    use_augmentation: bool = True
    normalize: bool = True
    seed: int = 42


class CIFAR10DataModule:
    """Builds dataloaders from a local train/test folder layout."""

    def __init__(self, config: Optional[CIFAR10DataConfig] = None) -> None:
        self.config = config or CIFAR10DataConfig()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.classes: Tuple[str, ...] = ()

    @property
    def train_dir(self) -> Path:
        return Path(self.config.data_dir) / "train"

    @property
    def test_dir(self) -> Path:
        return Path(self.config.data_dir) / "test"

    @property
    def manifest_path(self) -> Optional[Path]:
        if self.config.manifest_path is None:
            return None
        return Path(self.config.manifest_path)

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
        """Validates the expected local dataset structure."""
        if self.manifest_path is not None:
            if not self.manifest_path.exists() or not self.manifest_path.is_file():
                raise FileNotFoundError(f"Manifest file not found: {self.manifest_path}")
            return

        missing_paths = [
            str(path)
            for path in (self.train_dir, self.test_dir)
            if not path.exists() or not path.is_dir()
        ]
        if missing_paths:
            raise FileNotFoundError(
                "Expected local dataset folders were not found: "
                + ", ".join(missing_paths)
            )

    def setup(self) -> None:
        """Creates train, validation, and test dataset splits from local folders."""
        train_transform, eval_transform = self._build_transforms()

        if self.manifest_path is not None:
            manifest_entries = _read_manifest_entries(self.manifest_path)
            train_entries = [entry for entry in manifest_entries if entry[0] == "train"]
            test_entries = [entry for entry in manifest_entries if entry[0] == "test"]
            classes = tuple(sorted({entry[1] for entry in manifest_entries}))
            class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}

            full_train_dataset = ManifestImageDataset(
                entries=train_entries,
                class_to_idx=class_to_idx,
                transform=train_transform,
            )
            val_source_dataset = ManifestImageDataset(
                entries=train_entries,
                class_to_idx=class_to_idx,
                transform=eval_transform,
            )
            test_dataset = ManifestImageDataset(
                entries=test_entries,
                class_to_idx=class_to_idx,
                transform=eval_transform,
            )
        else:
            full_train_dataset = datasets.ImageFolder(
                root=str(self.train_dir),
                transform=train_transform,
            )

            val_source_dataset = datasets.ImageFolder(
                root=str(self.train_dir),
                transform=eval_transform,
            )

            test_dataset = datasets.ImageFolder(
                root=str(self.test_dir),
                transform=eval_transform,
            )

        train_classes = tuple(full_train_dataset.classes)
        test_classes = tuple(test_dataset.classes)
        if train_classes != test_classes:
            raise ValueError(
                "Train and test class folders do not match. "
                f"train={train_classes}, test={test_classes}"
            )
        self.classes = train_classes

        val_size = int(len(full_train_dataset) * self.config.validation_split)
        train_size = len(full_train_dataset) - val_size
        if val_size == 0:
            raise ValueError("validation_split is too small and produced an empty set.")

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


class ManifestImageDataset(Dataset):
    """Dataset backed by a manifest file containing split, class, and path columns."""

    def __init__(
        self,
        entries: list[tuple[str, str, str]],
        class_to_idx: dict[str, int],
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.entries = entries
        self.class_to_idx = class_to_idx
        self.classes = tuple(sorted(class_to_idx, key=class_to_idx.get))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        _, class_name, image_path = self.entries[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, self.class_to_idx[class_name]


def _read_manifest_entries(manifest_path: Path) -> list[tuple[str, str, str]]:
    lines = manifest_path.read_text().splitlines()
    if not lines:
        raise ValueError(f"Manifest is empty: {manifest_path}")

    entries: list[tuple[str, str, str]] = []
    for line in lines[1:]:
        if not line.strip():
            continue
        split, class_name, path = line.split("\t", maxsplit=2)
        entries.append((split, class_name, path))
    return entries
