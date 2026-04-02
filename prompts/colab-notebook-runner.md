# Colab Notebook Prompt

Date: 2026-04-02
Purpose: Generate a Google Colab notebook that can run the full project pipeline and display all outputs.

## Prompt
Create a Google Colab notebook for a CIFAR-10 CNN vs transformer project.

The notebook should:
- install required packages before importing project code
- work from the repository root
- run the full experiment pipeline from Python scripts already present in the repo
- allow the user to choose between:
  - full dataset
  - 50% subset
  - 10% subset
- choose the dataset subset through a manifest file option
- allow the user to set:
  - number of epochs
  - batch size
  - sample index for explainability output
  - dataset root path

The notebook should:
- call the main experiment runner script
- display the generated markdown report
- display the JSON results
- display plots for:
  - training and validation loss
  - validation accuracy
  - learning rate decay
  - model efficiency comparison
- display the saved explainability images for both models

Also ensure:
- imports happen before use
- package installation is done inside the notebook
- the notebook is easy to edit and rerun in Colab

## Notes
- Intended for end-to-end project execution and presentation.
- The notebook should be a convenient wrapper around the existing Python scripts rather than reimplementing the full project logic.
