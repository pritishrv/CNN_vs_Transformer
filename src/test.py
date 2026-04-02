import argparse

import torch
from torch import nn

from src.dataloader import CIFAR10DataConfig, CIFAR10DataModule
from src.utils import load_model_from_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained CIFAR-10 model.")
    parser.add_argument(
        "--model",
        choices=("cnn", "vit"),
        required=True,
        help="Model architecture to evaluate.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to the model checkpoint. Defaults to checkpoints/<model>.pth.",
    )
    return parser.parse_args()


def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            predictions = outputs.argmax(dim=1)
            total_correct += (predictions == labels).sum().item()
            total_examples += labels.size(0)

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = args.checkpoint or f"./checkpoints/{args.model}.pth"

    model, config, checkpoint = load_model_from_checkpoint(
        model_name=args.model,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    data_config = CIFAR10DataConfig(
        data_dir=config.data_dir,
        manifest_path=config.manifest_path,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        validation_split=config.validation_split,
        image_size=config.image_size,
        use_augmentation=False,
        normalize=True,
        seed=config.seed,
    )
    data_module = CIFAR10DataModule(data_config)
    data_module.prepare_data()
    data_module.setup()

    criterion = nn.CrossEntropyLoss()
    test_metrics = evaluate(
        model=model,
        dataloader=data_module.test_dataloader(),
        criterion=criterion,
        device=device,
    )

    best_val_accuracy = checkpoint.get("best_val_accuracy")
    best_val_text = "n/a" if best_val_accuracy is None else f"{best_val_accuracy:.4f}"
    print(
        f"Checkpoint: {checkpoint_path} | "
        f"best_val_acc={best_val_text} | "
        f"test_loss={test_metrics['loss']:.4f} | "
        f"test_acc={test_metrics['accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
