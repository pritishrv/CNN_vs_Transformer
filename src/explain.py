import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from src.dataloader import CIFAR10DataConfig, CIFAR10DataModule
from src.models import CIFAR10CNN, ViTLite
from src.utils import load_model_from_checkpoint

CIFAR10_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
CIFAR10_STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate saliency or attention visualisations for CIFAR-10."
    )
    parser.add_argument(
        "--model",
        choices=("cnn", "vit"),
        required=True,
        help="Model architecture to explain.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to the trained checkpoint. Defaults to checkpoints/<model>.pth.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Index of the test sample to visualise.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the output figure.",
    )
    return parser.parse_args()


def denormalize_image(image: torch.Tensor) -> torch.Tensor:
    return (image * CIFAR10_STD + CIFAR10_MEAN).clamp(0.0, 1.0)


def load_sample(config: CIFAR10DataConfig, sample_index: int) -> tuple[torch.Tensor, int]:
    data_module = CIFAR10DataModule(config)
    data_module.prepare_data()
    data_module.setup()
    return data_module.test_dataset[sample_index]


def load_class_names(config: CIFAR10DataConfig) -> tuple[str, ...]:
    data_module = CIFAR10DataModule(config)
    data_module.prepare_data()
    data_module.setup()
    return data_module.get_class_names()


def generate_saliency_map(model: CIFAR10CNN, image: torch.Tensor) -> tuple[int, torch.Tensor]:
    image = image.unsqueeze(0)
    image.requires_grad_(True)

    logits = model(image)
    predicted_class = logits.argmax(dim=1).item()
    score = logits[0, predicted_class]
    score.backward()

    saliency = image.grad.abs().max(dim=1).values.squeeze(0)
    saliency = saliency / (saliency.max() + 1e-8)
    return predicted_class, saliency.detach().cpu()


def generate_attention_map(model: ViTLite, image: torch.Tensor) -> tuple[int, torch.Tensor]:
    image = image.unsqueeze(0)

    with torch.no_grad():
        logits, _ = model(image, return_attention=True)
        predicted_class = logits.argmax(dim=1).item()
        attention_map = model.get_attention_map(image).squeeze(0)

    attention_map = F.interpolate(
        attention_map.unsqueeze(0).unsqueeze(0),
        size=(model.image_size, model.image_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)
    attention_map = attention_map / (attention_map.max() + 1e-8)
    return predicted_class, attention_map.cpu()


def save_visualisation(
    image: torch.Tensor,
    explanation_map: torch.Tensor,
    true_label: str,
    predicted_label: str,
    explanation_title: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(image.permute(1, 2, 0).numpy())
    axes[0].set_title(f"Original\nTrue: {true_label}")
    axes[0].axis("off")

    axes[1].imshow(image.permute(1, 2, 0).numpy())
    axes[1].imshow(explanation_map.numpy(), cmap="jet", alpha=0.5)
    axes[1].set_title(f"{explanation_title}\nPred: {predicted_label}")
    axes[1].axis("off")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = args.checkpoint or f"./checkpoints/{args.model}.pth"

    model, config, _ = load_model_from_checkpoint(
        model_name=args.model,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    data_config = CIFAR10DataConfig(
        data_dir=config.data_dir,
        batch_size=1,
        num_workers=config.num_workers,
        validation_split=config.validation_split,
        image_size=config.image_size,
        use_augmentation=False,
        normalize=True,
        seed=config.seed,
    )
    image, label = load_sample(data_config, args.sample_index)
    class_names = load_class_names(data_config)
    image = image.to(device)

    if args.model == "cnn":
        predicted_class, explanation_map = generate_saliency_map(model, image)
        title = "CNN Saliency"
    else:
        predicted_class, explanation_map = generate_attention_map(model, image)
        title = "Transformer Attention"

    output_path = Path(args.output or f"./outputs/{args.model}_sample_{args.sample_index}.png")
    save_visualisation(
        image=denormalize_image(image.detach().cpu()),
        explanation_map=explanation_map,
        true_label=class_names[label],
        predicted_label=class_names[predicted_class],
        explanation_title=title,
        output_path=output_path,
    )
    print(f"Saved explanation to {output_path}")


if __name__ == "__main__":
    main()
