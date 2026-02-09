# Robust Handwritten Digit Classification: Generalization in Deep Networks

**Module:** 7CCSMPNN: Pattern Recognition, Neural Networks and Deep Learning
**Project Type:** Comparative Analysis of Neural Architectures (MLP vs. CNN)
**Status:** [Active/Completed]

---

## 1. Executive Summary
This project investigates the challenge of **domain adaptation** and **generalization** in deep learning. While achieving high accuracy on the standard MNIST dataset is a solved problem, maintaining that performance on "wild," unseen handwriting (with different strokes, thickness, and orientations) remains a challenge.

This repository implements and compares two distinct architectures to solve this problem:
1.  **Multilayer Perceptron (MLP):** A dense baseline that treats images as flat vectors.
2.  **Convolutional Neural Network (CNN):** A deep architecture leveraging spatial invariance (local connectivity and pooling) to maximize performance on unseen data.

**Key Objective:** Achieve >97% accuracy on a private, unseen test set by mitigating overfitting and enforcing robust feature extraction.

---

## 2. Hypothesis & Methodology

### 2.1. The Challenge: Generalization
A model trained solely on MNIST often fails on real-world digits due to the **covariate shift**—the distribution of the training data (clean, centered MNIST) differs from the test data (varied handwriting styles).

### 2.2. Architectural Comparison

| Feature | Multilayer Perceptron (MLP) | Convolutional Neural Network (CNN) |
| :--- | :--- | :--- |
| **Input Handling** | Flattens 2D image to 1D vector ($28 \times 28 \to 784$) | Preserves 2D topology ($28 \times 28 \times 1$) |
| **Spatial Awareness** | **None:** A pixel at (0,0) is as related to (0,1) as it is to (27,27). | **High:** Convolution filters capture local patterns (edges, curves). |
| **Invariance** | Sensitive to small shifts/rotations. | **Translation Invariant** due to pooling layers. |
| **Hypothesis** | High training accuracy, but brittle on unseen/shifted data. | Superior generalization to "wild" data. |

### 2.3. Strategy
To bridge the generalization gap, this project employs:
* **Data Augmentation:** Random rotations ($\pm 10^\circ$), zooms, and shifts during training to simulate the "wild" dataset.
* **Regularization:** Dropout layers (rate 0.2 - 0.5) to prevent neuron co-adaptation.
* **Early Stopping:** Monitoring validation loss to prevent overfitting to the MNIST training set.

---
