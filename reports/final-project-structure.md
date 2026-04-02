# Final Report on Project Structure

## Overview

This report summarises the final structure of the project, with a focus on the code organisation, script responsibilities, and dataloader workflow.

The project compares a CNN and a Vision Transformer on CIFAR-10 using a shared PyTorch pipeline. It includes training, evaluation, explainability, and experiment comparison tools. The codebase was designed to work with a locally stored dataset and to remain practical for coursework-scale experiments.

## High-Level Structure

The repository is organised into a few main areas:

- `src/`
  Contains the Python source code for data loading, model definitions, training, evaluation, explainability, utilities, and experiment orchestration.

- `src/models/`
  Contains the two model definitions used in the project:
  - `cnn.py`
  - `vit.py`

- `reports/`
  Stores written documentation and report-style project notes.

- `prompts/`
  Stores reusable prompts used while building the project.

- `manifests/`
  Stores dataset manifest files for full, 50%, and 10% runs.

- `notebooks/`
  Stores the Google Colab notebook used to run the experiment workflow interactively.

## Core Script Responsibilities

### `src/config.py`

This file defines the shared training configuration. It centralises project settings such as:

- batch size
- learning rate
- learning rate decay settings
- number of epochs
- dataset root path
- optional manifest path
- checkpoint directory

This makes the pipeline easier to control and keeps script behaviour consistent.

### `src/dataloader.py`

This is one of the most important files in the project. It provides the data pipeline for training, validation, testing, and explainability.

The dataloader supports two modes:

1. Direct folder loading
   - Reads `train/` and `test/` folders from the dataset root.
   - Uses `torchvision.datasets.ImageFolder`.

2. Manifest-based loading
   - Reads image paths from a manifest txt file.
   - Uses a custom dataset class.
   - Supports experiments on full, 50%, or 10% dataset subsets.

The dataloader:

- applies data augmentation to training data
- applies evaluation transforms to validation and test data
- creates validation data from the training split
- infers class names from the dataset itself
- loads images lazily from disk

This design avoids loading the full dataset into RAM and is better suited to constrained environments.

## Why the Dataloader Design Matters

The project needed to work with a local folder-based CIFAR-10 dataset rather than the torchvision-downloaded version. That required a custom approach.

The final dataloader design solves several practical needs:

- local dataset support
- subset control through manifests
- batchwise loading from disk
- efficient use of RAM and VRAM
- compatibility with both CNN and transformer training
- reuse across training, testing, and explainability scripts

Only the current batch is sent to the GPU during training. This is important because it keeps VRAM usage controlled while still allowing CUDA-based model training.

## Model Files

### `src/models/cnn.py`

This file contains a compact CNN baseline for CIFAR-10. It follows a straightforward architecture with three convolution blocks and a classifier head. The model is intended to act as a strong baseline with simple and interpretable local feature extraction.

### `src/models/vit.py`

This file contains the lightweight Vision Transformer used in the comparison.

It includes:

- patch embedding
- positional embeddings
- transformer encoder blocks
- classification head
- attention extraction support

The attention support is important because it allows the project to generate attention maps for explainability analysis.

## Training and Evaluation Scripts

### `src/train.py`

This script trains a single selected model, either CNN or ViT. It:

- loads data
- trains the model epoch by epoch
- tracks validation performance
- saves the best checkpoint
- applies learning rate decay

It is useful when training only one model at a time.

### `src/test.py`

This script loads a saved checkpoint and evaluates the model on the test set. It provides final accuracy and loss numbers for a chosen trained model.

### `src/run_experiment.py`

This is the main comparison script for the whole project. It runs the full experiment workflow:

- trains both models
- tracks validation performance
- evaluates on the test set
- measures training time
- measures inference speed
- saves result summaries
- generates comparison plots
- generates explainability outputs

This file acts as the central experiment orchestrator.

## Explainability Script

### `src/explain.py`

This script generates visual explanations for trained models.

It supports:

- saliency maps for the CNN
- attention maps for the Vision Transformer

This is important because the project is not only about classification accuracy, but also about understanding what the models focus on when making decisions.

## Manifest Generation

### `src/generate_manifests.py`

This script scans the local dataset and creates three manifest files:

- full dataset
- half dataset
- 10% dataset

The subsets are sampled class by class so that class balance is preserved. This makes it easier to compare how the models behave under reduced data conditions.

## Notebook Support

### `notebooks/run_experiment_colab.ipynb`

The Colab notebook provides an interactive way to run the project.

It:

- installs dependencies
- allows dataset option selection
- chooses a manifest file
- runs the full experiment script
- displays reports and plots
- shows explainability outputs

This is useful for demonstration, reproducibility, and coursework presentation.

## Conclusion

The final project structure is organised around a clean experiment pipeline:

- flexible data loading
- reusable model definitions
- single-model and full-experiment training options
- evaluation and explainability support
- local-dataset compatibility
- Colab execution support

The most important technical decision in the final structure is the custom dataloader design. It makes the rest of the project practical by allowing local file-based training, subset selection through manifests, and efficient memory usage during experiments.
