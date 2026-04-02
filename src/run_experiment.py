import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from src.config import TrainingConfig
from src.dataloader import CIFAR10DataConfig, CIFAR10DataModule
from src.explain import (
    denormalize_image,
    generate_attention_map,
    generate_saliency_map,
    load_class_names,
    load_sample,
    save_visualisation,
)
from src.test import evaluate
from src.train import run_epoch, save_checkpoint, set_seed
from src.utils import build_model, load_model_from_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train, evaluate, compare, and explain CNN and ViT models on CIFAR-10."
    )
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs.")
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Training and evaluation batch size."
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
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Test sample index to use for explainability outputs.",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default="./reports",
        help="Directory to store experiment summaries.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("cnn", "vit"),
        default=("cnn", "vit"),
        help="Models to include in the run.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> TrainingConfig:
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


def build_data_module(config: TrainingConfig, use_augmentation: bool) -> CIFAR10DataModule:
    data_config = CIFAR10DataConfig(
        data_dir=config.data_dir,
        manifest_path=config.manifest_path,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        validation_split=config.validation_split,
        image_size=config.image_size,
        use_augmentation=use_augmentation,
        normalize=True,
        seed=config.seed,
    )
    data_module = CIFAR10DataModule(data_config)
    data_module.prepare_data()
    data_module.setup()
    return data_module


def now() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def measure_inference_time(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_examples = 0
    start_time = now()

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            _ = model(images)
            total_examples += images.size(0)

    total_time = now() - start_time
    return {
        "total_seconds": total_time,
        "seconds_per_sample": total_time / total_examples,
        "samples_per_second": total_examples / total_time,
    }


def train_and_evaluate_model(
    model_name: str,
    config: TrainingConfig,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    model = build_model(model_name, config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    scheduler = StepLR(
        optimizer,
        step_size=config.lr_decay_step,
        gamma=config.lr_decay_gamma,
    )

    best_val_accuracy = -1.0
    best_val_loss = 0.0
    best_checkpoint_path: Path | None = None
    history: list[dict[str, float | int]] = []

    train_start_time = now()
    for epoch in range(1, config.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            progress_label=f"{model_name} train {epoch}/{config.epochs}",
        )
        val_metrics = run_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            progress_label=f"{model_name} val {epoch}/{config.epochs}",
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            best_val_loss = val_metrics["loss"]
            best_checkpoint_path = save_checkpoint(
                model=model,
                model_name=model_name,
                config=config,
                best_val_accuracy=best_val_accuracy,
            )

        print(
            f"[{model_name}] epoch {epoch}/{config.epochs} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"train_acc={train_metrics['accuracy']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.6f}"
        )
        scheduler.step()

    total_train_time = now() - train_start_time

    if best_checkpoint_path is None:
        raise RuntimeError(f"No checkpoint saved for model {model_name}.")

    best_model, _, _ = load_model_from_checkpoint(
        model_name=model_name,
        checkpoint_path=best_checkpoint_path,
        device=device,
    )
    test_metrics = evaluate(
        model=best_model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
    )
    inference_timing = measure_inference_time(
        model=best_model,
        dataloader=test_loader,
        device=device,
    )

    return {
        "model": model_name,
        "checkpoint_path": str(best_checkpoint_path),
        "training_time_seconds": total_train_time,
        "best_validation": {
            "accuracy": best_val_accuracy,
            "loss": best_val_loss,
        },
        "test": test_metrics,
        "inference": inference_timing,
        "history": history,
    }


def generate_explanation(
    model_name: str,
    checkpoint_path: str,
    config: TrainingConfig,
    sample_index: int,
    device: torch.device,
) -> str:
    model, _, _ = load_model_from_checkpoint(
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    sample_config = CIFAR10DataConfig(
        data_dir=config.data_dir,
        batch_size=1,
        num_workers=config.num_workers,
        validation_split=config.validation_split,
        image_size=config.image_size,
        use_augmentation=False,
        normalize=True,
        seed=config.seed,
    )
    image, label = load_sample(sample_config, sample_index)
    image = image.to(device)

    if model_name == "cnn":
        predicted_class, explanation_map = generate_saliency_map(model, image)
        title = "CNN Saliency"
    else:
        predicted_class, explanation_map = generate_attention_map(model, image)
        title = "Transformer Attention"

    class_names = load_class_names(sample_config)
    output_path = Path(f"./outputs/{model_name}_sample_{sample_index}.png")
    save_visualisation(
        image=denormalize_image(image.detach().cpu()),
        explanation_map=explanation_map,
        true_label=class_names[label],
        predicted_label=class_names[predicted_class],
        explanation_title=title,
        output_path=output_path,
    )
    return str(output_path)


def build_markdown_report(results: list[dict[str, Any]]) -> str:
    lines = [
        "# Experiment Comparison Report",
        "",
        "| Model | Best Val Acc | Best Val Loss | Test Acc | Test Loss | Train Time (s) | Inference Time (s) | Sec / Sample | Samples / Sec |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for result in results:
        lines.append(
            "| "
            f"{result['model']} | "
            f"{result['best_validation']['accuracy']:.4f} | "
            f"{result['best_validation']['loss']:.4f} | "
            f"{result['test']['accuracy']:.4f} | "
            f"{result['test']['loss']:.4f} | "
            f"{result['training_time_seconds']:.2f} | "
            f"{result['inference']['total_seconds']:.4f} | "
            f"{result['inference']['seconds_per_sample']:.6f} | "
            f"{result['inference']['samples_per_second']:.2f} |"
        )

    return "\n".join(lines) + "\n"


def save_history_plots(results: list[dict[str, Any]], report_dir: Path) -> list[str]:
    output_paths: list[str] = []

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for result in results:
        epochs = [entry["epoch"] for entry in result["history"]]
        train_loss = [entry["train_loss"] for entry in result["history"]]
        val_loss = [entry["val_loss"] for entry in result["history"]]
        val_accuracy = [entry["val_accuracy"] for entry in result["history"]]
        learning_rates = [entry["learning_rate"] for entry in result["history"]]

        axes[0].plot(epochs, train_loss, label=f"{result['model']} train")
        axes[0].plot(epochs, val_loss, linestyle="--", label=f"{result['model']} val")
        axes[1].plot(epochs, val_accuracy, label=result["model"])
        axes[2].plot(epochs, learning_rates, label=result["model"])

    axes[0].set_title("Training and Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    axes[2].set_title("Learning Rate Decay")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning Rate")
    axes[2].legend()

    fig.tight_layout()
    history_plot_path = report_dir / "training_validation_curves.png"
    fig.savefig(history_plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    output_paths.append(str(history_plot_path))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    model_names = [result["model"] for result in results]
    val_accuracies = [result["best_validation"]["accuracy"] for result in results]
    train_times = [result["training_time_seconds"] for result in results]
    inference_rates = [result["inference"]["samples_per_second"] for result in results]

    axes[0].bar(model_names, val_accuracies, color=["#1f77b4", "#ff7f0e"][: len(results)])
    axes[0].set_title("Best Validation Accuracy")
    axes[0].set_ylabel("Accuracy")

    width = 0.35
    x_positions = list(range(len(model_names)))
    axes[1].bar(
        [x - width / 2 for x in x_positions],
        train_times,
        width=width,
        label="Train time (s)",
    )
    axes[1].bar(
        [x + width / 2 for x in x_positions],
        inference_rates,
        width=width,
        label="Samples / sec",
    )
    axes[1].set_xticks(x_positions, model_names)
    axes[1].set_title("Efficiency Comparison")
    axes[1].legend()

    fig.tight_layout()
    comparison_plot_path = report_dir / "comparison_metrics.png"
    fig.savefig(comparison_plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    output_paths.append(str(comparison_plot_path))

    return output_paths


def main() -> None:
    args = parse_args()
    config = build_config(args)
    set_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data_module = build_data_module(config, use_augmentation=True)
    eval_data_module = build_data_module(config, use_augmentation=False)

    train_loader = train_data_module.train_dataloader()
    val_loader = eval_data_module.val_dataloader()
    test_loader = eval_data_module.test_dataloader()

    results: list[dict[str, Any]] = []
    for model_name in args.models:
        result = train_and_evaluate_model(
            model_name=model_name,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
        )
        result["explainability_output"] = generate_explanation(
            model_name=model_name,
            checkpoint_path=result["checkpoint_path"],
            config=config,
            sample_index=args.sample_index,
            device=device,
        )
        results.append(result)

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    json_report_path = report_dir / "experiment_results.json"
    markdown_report_path = report_dir / "experiment_results.md"
    plot_paths = save_history_plots(results, report_dir)

    summary_payload = {
        "results": results,
        "plots": plot_paths,
    }

    json_report_path.write_text(json.dumps(summary_payload, indent=2))
    markdown_report_path.write_text(build_markdown_report(results))

    print(f"Saved JSON report to {json_report_path}")
    print(f"Saved Markdown report to {markdown_report_path}")
    for plot_path in plot_paths:
        print(f"Saved plot to {plot_path}")
    for result in results:
        print(
            f"{result['model']} | "
            f"val_acc={result['best_validation']['accuracy']:.4f} | "
            f"test_acc={result['test']['accuracy']:.4f} | "
            f"train_time={result['training_time_seconds']:.2f}s | "
            f"inference_total={result['inference']['total_seconds']:.4f}s | "
            f"explain={result['explainability_output']}"
        )


if __name__ == "__main__":
    main()
