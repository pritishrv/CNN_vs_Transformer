# Dataset Research Prompt

Date: 2026-04-01
Purpose: Ask an LLM to identify a simple image dataset with around 10 classes for model training, comparison, and explainability work.

## Prompt
Research and recommend a simple image classification dataset suitable for a university project.

The dataset should meet these requirements:
- roughly 10 classes or close to that number
- suitable for supervised image classification
- simple enough to train both a CNN and a transformer model without requiring very large compute
- appropriate for comparing model performance
- appropriate for explainability analysis using saliency maps and attention maps
- not overly complicated in terms of image content, preprocessing, or access

The project goal is to:
- train a CNN model
- train a transformer-based vision model
- compare their performance
- compare their explainability using saliency and attention-based visualisations

Please provide:
1. 3 to 5 dataset recommendations
2. for each dataset, include number of classes, image type, approximate dataset size, difficulty level, and why it is suitable
3. a short comparison of the recommended datasets
4. your best final recommendation for this project
5. any practical concerns such as class imbalance, download availability, image resolution, or licensing

Prioritise datasets that are easy to access, well known, and realistic for a small-to-medium deep learning project.

## Notes
- Intended for early project scoping.
- The preferred outcome is a dataset that is simple, balanced, and easy to use for both baseline CNNs and vision transformers.
