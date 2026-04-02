# CNN vs Transformer on CIFAR-10

This project compares a Convolutional Neural Network (CNN) and a lightweight Vision Transformer (ViT-lite) on CIFAR-10. The goal is to benchmark their performance, training behaviour, efficiency, and explainability using a common dataset and a shared training pipeline.

The project supports:
- local folder-based CIFAR-10 data
- manifest-based full, 50%, and 10% dataset runs
- CNN and transformer training
- validation and test evaluation
- timing comparison
- saliency and attention-based explainability
- a Colab notebook runner

## Project Goal

The main research question is:

Do transformers capture more meaningful global patterns than CNNs on the same image classification task, and how does that affect both performance and interpretability?

## Dataset Layout

The code is designed for a local dataset with this structure:

```text
cifar10/
  train/
    airplane/
    automobile/
    bird/
    ...
  test/
    airplane/
    automobile/
    bird/
    ...
```

The training pipeline can also use manifest files that list selected image paths for:
- full dataset
- 50% subset
- 10% subset

Generated manifests are stored in `manifests/`.

## Project Structure

```text
CNN_vs_Transformer/
  manifests/
  notebooks/
  project-context/
  prompts/
  reports/
  src/
    models/
  README.md
  WORK_DIARY.md
  requirements.txt
```

## Main Files

### Core Code

- `src/config.py`
  Central training configuration including batch size, learning rate, decay settings, dataset path, and manifest path.

- `src/dataloader.py`
  Custom dataloader module supporting both:
  - direct folder-based loading with `ImageFolder`
  - manifest-based loading through a custom dataset class

- `src/models/cnn.py`
  CNN baseline for CIFAR-10.

- `src/models/vit.py`
  Lightweight Vision Transformer for CIFAR-10 with attention-map support.

- `src/train.py`
  Train a single model with validation tracking and learning rate decay.

- `src/test.py`
  Load a trained checkpoint and evaluate it on the test set.

- `src/explain.py`
  Generate explainability outputs:
  - saliency maps for CNN
  - attention maps for ViT

- `src/run_experiment.py`
  End-to-end experiment runner that:
  - trains both models
  - evaluates both models
  - measures time and inference speed
  - saves comparison reports and plots
  - generates explainability outputs

- `src/generate_manifests.py`
  Create full, 50%, and 10% manifest txt files from the local dataset.

### Documentation and Reports

- `WORK_DIARY.md`
  Running log of project work.

- `prompts/`
  Reusable prompts used to help build parts of the project.

- `reports/`
  Dataset notes and final written summaries.

- `notebooks/run_experiment_colab.ipynb`
  Colab notebook to run the full experiment pipeline.

## Environment Setup

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Generating Dataset Manifests

If your dataset is stored locally, generate manifests with:

```bash
python -m src.generate_manifests --data-dir "/Users/pritishrv/Documents/Courseworks/NeuralComputing/archive (5)/cifar10"
```

This produces:
- `manifests/cifar10_full.txt`
- `manifests/cifar10_half.txt`
- `manifests/cifar10_ten_percent.txt`

## Running Training

Train one model directly:

```bash
python -m src.train --model cnn --data-dir "/Users/pritishrv/Documents/Courseworks/NeuralComputing/archive (5)/cifar10"
python -m src.train --model vit --data-dir "/Users/pritishrv/Documents/Courseworks/NeuralComputing/archive (5)/cifar10"
```

To train using a manifest subset:

```bash
python -m src.train --model cnn \
  --data-dir "/Users/pritishrv/Documents/Courseworks/NeuralComputing/archive (5)/cifar10" \
  --manifest-path manifests/cifar10_half.txt
```

## Running the Full Comparison

Run the complete experiment:

```bash
python -m src.run_experiment \
  --data-dir "/Users/pritishrv/Documents/Courseworks/NeuralComputing/archive (5)/cifar10" \
  --manifest-path manifests/cifar10_full.txt
```

This will:
- train CNN and ViT
- save best checkpoints
- evaluate test performance
- measure training time and inference speed
- save report files and plots
- generate explainability images

## Outputs

Typical outputs are:
- `checkpoints/`
- `outputs/`
- `reports/experiment_results.md`
- `reports/experiment_results.json`
- `reports/training_validation_curves.png`
- `reports/comparison_metrics.png`

## Colab Notebook

The notebook:
- installs packages before use
- allows choosing `full`, `half`, or `ten_percent`
- runs the experiment runner
- displays reports, plots, and explainability outputs

File:
- `notebooks/run_experiment_colab.ipynb`

## Notes on Memory Usage

The dataset pipeline is designed to be memory-aware:
- images are read from disk lazily
- the full dataset is not loaded into RAM
- data is fed to the model batch by batch
- only the current batch is moved to CUDA during training

## Reports

Additional project write-ups are in:
- `reports/report-cifar10.md`
- `reports/report-dataset.md`
- `reports/final-project-structure.md`
