import argparse
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from src.config import TrainingConfig
from src.dataloader import CIFAR10DataConfig, CIFAR10DataModule
from src.utils import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CNN or ViT on CIFAR-10.")
    parser.add_argument(
        "--model",
        choices=("cnn", "vit"),
        required=True,
        help="Model architecture to train.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs.")
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Training batch size."
    )
    parser.add_argument(
        "--learning-rate", type=float, default=None, help="Optimizer learning rate."
    )
    parser.add_argument(
        "--lr-decay-step",
        type=int,
        default=None,
        help="Number of epochs between learning rate decay steps.",
    )
    parser.add_argument(
        "--lr-decay-gamma",
        type=float,
        default=None,
        help="Multiplicative factor applied during learning rate decay.",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None, help="Directory for CIFAR-10 data."
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        default=None,
        help="Optional manifest txt file describing dataset paths to use.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory to store trained checkpoints.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of dataloader worker processes.",
    )
    return parser.parse_args()


def build_training_config(args: argparse.Namespace) -> TrainingConfig:
    config = TrainingConfig()
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.lr_decay_step is not None:
        config.lr_decay_step = args.lr_decay_step
    if args.lr_decay_gamma is not None:
        config.lr_decay_gamma = args.lr_decay_gamma
    if args.data_dir is not None:
        config.data_dir = args.data_dir
    if args.manifest_path is not None:
        config.manifest_path = args.manifest_path
    if args.checkpoint_dir is not None:
        config.checkpoint_dir = args.checkpoint_dir
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    return config


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataloaders(config: TrainingConfig) -> Tuple[torch.utils.data.DataLoader, ...]:
    data_config = CIFAR10DataConfig(
        data_dir=config.data_dir,
        manifest_path=config.manifest_path,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        validation_split=config.validation_split,
        image_size=config.image_size,
        use_augmentation=True,
        normalize=True,
        seed=config.seed,
    )
    data_module = CIFAR10DataModule(data_config)
    data_module.prepare_data()
    data_module.setup()
    return (
        data_module.train_dataloader(),
        data_module.val_dataloader(),
        data_module.test_dataloader(),
    )


def run_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Adam | None = None,
) -> Dict[str, float]:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            if is_training:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            predictions = outputs.argmax(dim=1)
            total_correct += (predictions == labels).sum().item()
            total_examples += labels.size(0)

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


def save_checkpoint(
    model: nn.Module,
    model_name: str,
    config: TrainingConfig,
    best_val_accuracy: float,
) -> Path:
    checkpoint_dir = config.checkpoint_path
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{model_name}.pth"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "best_val_accuracy": best_val_accuracy,
            "config": config.__dict__,
        },
        checkpoint_path,
    )
    return checkpoint_path


def main() -> None:
    args = parse_args()
    config = build_training_config(args)
    set_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = build_dataloaders(config)
    model = build_model(args.model, config).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    scheduler = StepLR(
        optimizer,
        step_size=config.lr_decay_step,
        gamma=config.lr_decay_gamma,
    )

    best_val_accuracy = 0.0
    for epoch in range(1, config.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )
        val_metrics = run_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            checkpoint_path = save_checkpoint(
                model=model,
                model_name=args.model,
                config=config,
                best_val_accuracy=best_val_accuracy,
            )
        else:
            checkpoint_path = config.checkpoint_path / f"{args.model}.pth"

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch}/{config.epochs} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"train_acc={train_metrics['accuracy']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"lr={current_lr:.6f} | "
            f"checkpoint={checkpoint_path}"
        )
        scheduler.step()

    test_metrics = run_epoch(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
    )
    print(
        f"Test results | loss={test_metrics['loss']:.4f} | "
        f"accuracy={test_metrics['accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
