# Custom Dataloader Prompt

Date: 2026-04-02
Purpose: Build a custom dataloader that works with a local CIFAR-10 dataset or manifest files while keeping RAM and VRAM usage efficient.

## Prompt
Create a custom PyTorch dataloader module for a CIFAR-10 project where the dataset is stored locally on disk and should not be loaded fully into RAM.

Requirements:
- support a folder-based dataset structure with `train/` and `test/` directories
- also support optional manifest txt files that list image paths, class names, and split names
- load images lazily from disk, one sample at a time, through the dataset and DataLoader pipeline
- return batches through a standard PyTorch DataLoader
- create train, validation, and test loaders
- create validation data from the training split
- use torchvision transforms for augmentation and normalization
- infer class names from folder names or the manifest file
- keep the design compatible with CNN and transformer training on CIFAR-10 images

Optimisation goals:
- do not preload the entire dataset into RAM
- keep batch loading disk-based
- ensure only the current batch is moved to CUDA during training
- make the dataloader suitable for efficient RAM and VRAM usage

Implementation details:
- use `ImageFolder` when reading directly from folders
- use a custom Dataset when reading from manifest files
- expose train, validation, and test DataLoader methods
- keep the interface reusable for training, testing, and explainability scripts

## Notes
- Intended for a project comparing CNN and transformer models.
- The dataloader should be easy to switch between full, half, and 10% manifest subsets.

