# CIFAR-10 Dataset Report

## Overview

CIFAR-10 is the dataset selected for this project. It is a well-known benchmark dataset for image classification and is widely used in deep learning experiments, especially for introductory and intermediate computer vision tasks.

The dataset contains 60,000 colour images of size 32x32 pixels. These images are divided into 10 classes, with 6,000 images per class. The standard split is 50,000 training images and 10,000 test images.

## Class Labels

The 10 CIFAR-10 classes are:
- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

## Why CIFAR-10 Fits This Project

CIFAR-10 is a strong choice for this project for several reasons:

- It has exactly 10 classes, which matches the intended scale of the study.
- It is simple and widely accessible, making it practical for a university project.
- It is large enough to support meaningful training and evaluation, but still manageable on modest hardware.
- It can be used to train both a CNN and a transformer-based vision model.
- It is suitable for comparing model behaviour, accuracy, and generalisation.
- It is also suitable for explainability analysis, including saliency maps and attention-based visualisations.

## Relevance to CNN and Transformer Comparison

CNNs have historically performed well on CIFAR-10 because the dataset is a standard benchmark for convolution-based image classification. The dataset is therefore useful for building a strong CNN baseline.

Transformer-based vision models can also be trained on CIFAR-10, although the small image size may require careful model design or preprocessing. This makes the dataset useful for comparing how the two model families behave under the same task and data conditions.

Using the same dataset for both models ensures that performance differences can be attributed more clearly to the model architectures rather than to differences in data.

## Relevance to Explainability

This project also aims to study explainability. CIFAR-10 supports this well because:

- saliency methods can be used to highlight which image regions influenced CNN predictions
- attention maps can be examined to understand what parts of the image a transformer focuses on
- the classes are visually distinct enough to make qualitative interpretation possible

Although the images are small, they are still sufficient for demonstrating and comparing explanation methods in a clear and manageable way.

## Practical Considerations

- The low image resolution of 32x32 makes training efficient.
- The dataset is balanced, with an equal number of images per class.
- CIFAR-10 is available directly through common deep learning libraries such as PyTorch and TensorFlow.
- Because it is a standard benchmark, it is easy to compare project results with existing reference results in the literature.

## Conclusion

CIFAR-10 is an appropriate dataset for this project because it is simple, balanced, well established, and computationally manageable. It provides a practical basis for training and comparing a CNN and a transformer model, while also supporting explainability analysis through saliency and attention visualisations.
