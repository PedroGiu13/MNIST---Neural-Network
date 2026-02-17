# Handwritten Digit Classification - MLP & CNN

## Project Overview
This project investigates the chanllenges of building and training an Artificial Neural Network for classifying handwritten digits. In particular, the project focuses on the differences between the classic Fully Connected Neural Network, the Multilayer Perceptron (MLP), and more advanced Deep Neural Networks, such as a Convolutional Neural Network (CNN).

Both neural netweworks where trained using the same data (MNIST dataset) and then tested in a completely new dataset (USPS dataset). The end goal of the project is to understand how and why deep neural networks excell at this type of classification problems, and why 'shallower' neural networks struggle. Although, it should be theoretically possible to approximate any continuous function using a "simple" three layer ANN.

---

## Components
### 1. Multilayer Perceptron (MLP)
**What is a Multilayer Perceptron?**
























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
