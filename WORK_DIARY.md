# Work Diary

This file records the work completed on this project.

## 2026-04-01

### Session Start
- Created the project work diary.
- Created the `prompts/` directory for storing selected prompts used during the project.
- Added prompt storage guidelines in `prompts/README.md`.
- Added a reusable dataset research prompt in `prompts/` for identifying a simple image dataset with around 10 classes for CNN vs transformer comparison and explainability analysis.
- Added a project `README.md` describing the intent and scope of the repository.
- Added a short report on CIFAR-10 as the intended dataset for the project.
- Created the `src/` folder and added a CIFAR-10 dataloader class for training, validation, and test splits.

## 2026-04-02

### Model Setup
- Added separate CNN and transformer model files under `src/models/` for CIFAR-10, based on the project context plan.
- Added shared training configuration and a training script to train either the CNN or the transformer on CIFAR-10.
- Added evaluation and explainability scripts, including checkpoint-loading utilities, CNN saliency maps, and transformer attention map generation.
- Added a `requirements.txt` environment file for project setup.
- Added a `.gitignore` to exclude local system files, datasets, checkpoints, outputs, and Python cache files.
