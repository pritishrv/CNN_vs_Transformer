import argparse
import random
from pathlib import Path

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate manifest txt files for a local folder-based CIFAR-10 dataset."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/Users/pritishrv/Documents/Courseworks/NeuralComputing/archive (5)/cifar10",
        help="Dataset root containing train/ and test/ folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./manifests",
        help="Directory where manifest txt files will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for subset selection.",
    )
    return parser.parse_args()


def list_split_entries(split_dir: Path) -> list[tuple[str, str, str]]:
    if not split_dir.exists() or not split_dir.is_dir():
        raise FileNotFoundError(f"Missing split directory: {split_dir}")

    entries: list[tuple[str, str, str]] = []
    class_dirs = sorted(path for path in split_dir.iterdir() if path.is_dir())
    for class_dir in class_dirs:
        image_paths = sorted(
            path
            for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
        )
        for image_path in image_paths:
            entries.append((split_dir.name, class_dir.name, str(image_path.resolve())))
    return entries


def sample_entries(
    entries: list[tuple[str, str, str]], fraction: float, seed: int
) -> list[tuple[str, str, str]]:
    if fraction >= 1.0:
        return entries

    grouped: dict[str, list[tuple[str, str, str]]] = {}
    for entry in entries:
        grouped.setdefault(entry[1], []).append(entry)

    rng = random.Random(seed)
    sampled: list[tuple[str, str, str]] = []
    for class_name in sorted(grouped):
        class_entries = list(grouped[class_name])
        rng.shuffle(class_entries)
        sample_size = max(1, int(len(class_entries) * fraction))
        sampled.extend(sorted(class_entries[:sample_size], key=lambda item: item[2]))

    return sampled


def build_manifest_lines(
    train_entries: list[tuple[str, str, str]],
    test_entries: list[tuple[str, str, str]],
) -> list[str]:
    lines = ["split\tclass_name\tpath"]
    lines.extend(f"{split}\t{class_name}\t{path}" for split, class_name, path in train_entries)
    lines.extend(f"{split}\t{class_name}\t{path}" for split, class_name, path in test_entries)
    return lines


def write_manifest(
    output_path: Path,
    train_entries: list[tuple[str, str, str]],
    test_entries: list[tuple[str, str, str]],
) -> None:
    lines = build_manifest_lines(train_entries, test_entries)
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    full_train_entries = list_split_entries(data_dir / "train")
    full_test_entries = list_split_entries(data_dir / "test")

    manifest_specs = [
        ("cifar10_full.txt", 1.0),
        ("cifar10_half.txt", 0.5),
        ("cifar10_ten_percent.txt", 0.1),
    ]

    for file_name, fraction in manifest_specs:
        train_entries = sample_entries(full_train_entries, fraction, args.seed)
        test_entries = sample_entries(full_test_entries, fraction, args.seed)
        output_path = output_dir / file_name
        write_manifest(output_path, train_entries, test_entries)
        print(
            f"Wrote {output_path} | "
            f"train={len(train_entries)} | test={len(test_entries)} | fraction={fraction}"
        )


if __name__ == "__main__":
    main()
