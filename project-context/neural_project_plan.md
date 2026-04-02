# Neural Computing Project — CNN vs Transformer with Explainability (PyTorch)

## 🎯 Project Overview

This project implements and compares:
- A Convolutional Neural Network (CNN)
- A Vision Transformer (ViT-lite)

on the CIFAR-10 dataset, followed by explainability analysis.

Goal:
Understand how different architectures learn and what they focus on.

---

## 📁 Project Structure

project/
│── data/
│── models/
│   ├── cnn.py
│   ├── vit.py
│── dataloader.py
│── train.py
│── test.py
│── explain.py
│── utils.py
│── config.py
│── requirements.txt

---

## ⚙️ Requirements

torch
torchvision
numpy
matplotlib
tqdm

---

## 📊 Dataset

CIFAR-10:
- 10 classes
- 32x32 RGB images

---

## 📦 dataloader.py

- Load CIFAR-10
- Apply transforms
- Return train/test loaders

Normalization:
mean = (0.5, 0.5, 0.5)
std = (0.5, 0.5, 0.5)

Function:
get_dataloaders(batch_size)

---

## 🧠 CNN Model

Architecture:
Conv → ReLU → Pool (x3)
Flatten → FC → ReLU → FC (10 classes)

Expected:
- Fast
- Good local feature learning
- ~65–75% accuracy

---

## 🧠 Transformer Model (ViT-lite)

Steps:
1. Patch embedding
2. Positional encoding
3. Transformer encoder
4. Classification head

Expected:
- Slower
- Captures global patterns
- ~60–80% accuracy

---

## ⚙️ train.py

- Train CNN or ViT
- Loss: CrossEntropyLoss
- Optimizer: Adam

Output:
checkpoints/cnn.pth
checkpoints/vit.pth

---

## 🧪 test.py

Evaluate:
- Accuracy
- Loss

---

## 🔍 explain.py

CNN:
- Saliency maps (gradient-based)

Transformer:
- Attention maps

Output:
Original vs CNN vs Transformer visualization

---

## ⚙️ config.py

BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 20
PATCH_SIZE = 4
EMBED_DIM = 128

---

## 🧪 Experiments

1. Accuracy comparison
2. Generalisation
3. Explainability
4. Failure analysis

---

## 🚀 Workflow

1. Load data
2. Train CNN
3. Train Transformer
4. Evaluate
5. Generate explanations
6. Analyse results

---

## 🎯 Key Question

Do transformers capture more meaningful global patterns than CNNs?
