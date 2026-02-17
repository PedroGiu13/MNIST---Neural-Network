# Handwritten Digit Classification - MLP vs. CNN

## Project Overview
This project investigates the chanllenges of building and training an Artificial Neural Network for classifying handwritten digits. In particular, the project focuses on the differences between the classic Fully Connected Neural Network, the Multilayer Perceptron (MLP), and more advanced Deep Neural Networks, such as a Convolutional Neural Network (CNN).

Both neural netweworks where trained using the same data (MNIST dataset) and then tested in a completely new dataset (USPS dataset). The end goal of the project is to understand how and why deep neural networks excell at this type of classification problems, and why 'shallower' neural networks struggle. Although, it should be theoretically possible to approximate any continuous function using a "simple" three layer ANN.

---

## Components
### 1. Multilayer Perceptron (MLP)
**What is a Multilayer Perceptron?**
An MLP is a type of Feed Forward Network consiting of one input layer, no more than 3 hidden layers and one output layer. Each layer has a set of Linear Threshold Units (neurons) that are interconnected by links, and each link has an associated connection weight. In addtion, each layer has a single Bias unit that is connected to all the units in the subsecuent layer. In theory, any three layer (input - hidden - output) MLP should be a Universal Approximator, meaning it is capable of modelling any non linear function arbitrarly well.

**Architecture**


**Training**


**Performance**

### 2. Convolutional Neural Network
**What is a Multilayer Perceptron?**
An MLP is a type of Feed Forward Network consiting of one input layer, no more than 3 hidden layers and one output layer. Each layer has a set of Linear Threshold Units (neurons) that are interconnected by links, and each link has an associated connection weight. In addtion, each layer has a single Bias unit that is connected to all the units in the subsecuent layer. In theory, any three layer (input - hidden - output) MLP should be a Universal Approximator, meaning it is capable of modelling any non linear function arbitrarly well.

**Architecture**


**Training**
- Hyperparameter Tunning
- Best model training
- Data Augmentation

| Accuracy | Loss |
| :---: | :---: |
| ![MLP Accuracy](img/mlp_model_accuracy.png) | ![MLP Loss](img/mlp_model_loss.png) |


**Performance**


---
## Performance in new domain






















