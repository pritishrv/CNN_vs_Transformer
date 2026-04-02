# Project Scripts Prompt

Date: 2026-04-02
Purpose: Generate the main script pipeline for CNN vs transformer training, evaluation, explainability, and comparison.

## Prompt
Create the core Python scripts for a university project that compares a CNN and a transformer-based vision model on CIFAR-10.

The project already needs:
- a custom dataloader
- a CNN model
- a lightweight vision transformer model
- support for explainability analysis

Please create or outline scripts for:
- `config.py`
- `train.py`
- `test.py`
- `explain.py`
- `run_experiment.py`
- `utils.py`

Project requirements:
- train either a CNN or a transformer model on CIFAR-10
- evaluate validation and test performance
- compare the two models directly
- record metrics such as:
  - training loss
  - validation loss
  - validation accuracy
  - test loss
  - test accuracy
  - total training time
  - inference speed or runtime
- include learning rate decay during training
- save checkpoints for the best validation performance
- generate explainability outputs:
  - saliency maps for the CNN
  - attention maps for the transformer
- save comparison reports and plots

The experiment runner should:
- train both models
- evaluate both models
- generate comparison plots
- save explainability outputs
- optionally use a manifest file for full, 50%, or 10% dataset runs

Keep the implementation practical, clear, and suitable for a coursework project.

## Notes
- Intended to generate the full training and evaluation pipeline.
- The output should be compatible with local-disk datasets and manifest-based subsets.

