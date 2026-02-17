# Handwritten Digit Classification - MLP vs. CNN

## Project Overview
This project investigates the chanllenges of building and training an Artificial Neural Network for classifying handwritten digits. In particular, the project focuses on the differences between the classic Fully Connected Neural Network, the Multilayer Perceptron (MLP), and more advanced Deep Neural Networks, such as a Convolutional Neural Network (CNN).

Both neural netweworks where trained using the same data (MNIST dataset) and then tested in a completely new dataset (USPS dataset). The end goal of the project is to understand how and why deep neural networks excell at this type of classification problems, and why 'shallower' neural networks struggle. Although, it should be theoretically possible to approximate any continuous function using a "simple" three layer ANN.

---
## Data

---
## Components
### 1. Multilayer Perceptron (MLP)

**- What is a Multilayer Perceptron?**

An MLP is a type of Feed Forward Network consiting of one input layer, no more than 3 hidden layers, and one output layer. Each layer has a set of Linear Threshold Units (neurons) that are interconnected by links, and each link has an associated connection weight. In addtion, each layer has a single Bias unit that is connected to all the units in the subsecuent layer. In theory, any three layer (input - hidden - output) MLP should be a Universal Approximator, meaning it is capable of modelling any non linear function arbitrarly well.

**- MLP Architecture**

Input (784) → Dense(224) → ReLU → Dropout → Dense(288) → ReLU → Dropout → Dense(10) → Softmax

| Layer | Type | Output Shape | Parameters |
|-------|------|-------------|------------|
| dense | Dense (224 units) | (224) | 175,840 |
| activation | ReLU | (224) | 0 |
| dropout | Dropout | (224) | 0 |
| dense_1 | Dense (288 units) | (288) | 64,800 |
| activation_1 | ReLU | (288) | 0 |
| dropout_1 | Dropout | (288) | 0 |
| dense_2 | Dense (10 units) | (10) | 2,890 |
| activation_2 | Softmax | (10) | 0 |

**Total Parameters:** 243,532  

**- Training**
- Hyperparameter Tunning
- Best model training
- Data Augmentation

| Accuracy | Loss |
| :---: | :---: |
| ![MLP Accuracy](img/mlp_model_accuracy.png) | ![MLP Loss](img/mlp_model_loss.png) |


**- Performance**
<div align="center">
  <img src="./img/mlp_model_cm.png" width="80%" alt="MLP confusion Matrix">
  <!-- <p><em>Figure 1: Visual representation of the model layers and tensor shapes.</em></p> -->
</div>

 
### 2. Convolutional Neural Network (CNN)
**- What is a Convolutional Neural Network?**

**- CNN Architecture**

Input → Conv2D(32) → MaxPool → Conv2D(192) → MaxPool → Flatten → Dense(128) → Dropout → Dense(10) → Softmax

| Layer | Type | Output Shape | Parameters |
|-------|------|-------------|------------|
| conv2d | Conv2D | (28, 28, 32) | 320 |
| max_pooling2d | MaxPooling2D | (14, 14, 32) | 0 |
| conv2d_1 | Conv2D | (14, 14, 192) | 55,488 |
| max_pooling2d_1 | MaxPooling2D | (7, 7, 192) | 0 |
| flatten | Flatten | (9,408) | 0 |
| dense | Dense (ReLU) | (128) | 1,204,352 |
| dropout | Dropout | (128) | 0 |
| dense_1 | Dense (Softmax) | (10) | 1,290 |

**Total Parameters:** 1,261,452  


**- Training**

| Accuracy | Loss |
| :---: | :---: |
| ![CNN Accuracy](img/cnn_model_accuracy.png) | ![CNN Loss](img/cnn_model_loss.png) |

**- Performance**
![CNN Confusion Matrix](img/cnn_model_cm.png)

---
## Performance in new domain






















