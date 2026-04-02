# Data Manifest Preparation Prompt

Date: 2026-04-02
Purpose: Generate manifest files for a local CIFAR-10 folder dataset so experiments can run on full, half, or 10% subsets without changing the dataset itself.

## Prompt
Create a Python script for a local CIFAR-10 dataset stored in this folder structure:

```text
cifar10/
  train/
    airplane/
    automobile/
    ...
  test/
    airplane/
    automobile/
    ...
```

The script should:
- scan the local dataset from disk
- create manifest txt files instead of copying image files
- write absolute image paths
- include split and class information for every image
- create 3 manifest files:
  - one using the full training and full testing dataset
  - one using 50% of the training and 50% of the testing dataset
  - one using 10% of the training and 10% of the testing dataset
- choose the 50% and 10% subsets randomly
- preserve class balance by sampling class by class
- use a fixed random seed for reproducibility

Each manifest file should use a simple tab-separated format like:
```text
split	class_name	path
train	airplane	/absolute/path/to/image.png
```

The script should be designed so the generated manifests can later be used by a custom PyTorch dataloader.

## Notes
- Intended for subset control without duplicating dataset files.
- The output should support lazy loading from disk during training.

