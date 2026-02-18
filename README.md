# Handwritten Digit Classification - MLP vs. CNN

## Project Overview
This project investigates the chanllenges of building and training an Artificial Neural Network for classifying handwritten digits. In particular, the project focuses on the differences between the classic Fully Connected Neural Network, the Multilayer Perceptron (MLP), and more advanced Deep Neural Networks, such as a Convolutional Neural Network (CNN).

Both neural netweworks where trained using the same data (MNIST dataset) and then tested in a completely new dataset (USPS dataset). The end goal of the project is to understand how and why deep neural networks excell at this type of classification problems, and why 'shallower' neural networks struggle. Although, it should be theoretically possible to approximate any continuous function using a "simple" three layer ANN.

---
## Data
This project uses the **MNIST dataset** of handwritten digits to train both neural networks.
* **Source:** [The MNIST Database](http://yann.lecun.com/exdb/mnist/)
* **Authors:** Yann LeCun, Corinna Cortes, Christopher J.C. Burges
* **Details:** The dataset contains 70,000 images divided into 60,000 training examples and 10,000 testing examples.

In addition, the both models are evaluated in with an external dataset to test how well both networks generalize the data.
* **Source:** [USPS Database](https://ieeexplore.ieee.org/document/291440/authors#authors)
* **Authors:** J. J. Hull
* **Details:** The dataset contains 9.298 images divided into 7.291 training examples and 2007 testing examples

---
## Components
### 1. Multilayer Perceptron (MLP)

**- What is a Multilayer Perceptron?**

An MLP is a type of Feed Forward Network consiting of one input layer, no more than 3 hidden layers, and one output layer. Each layer has a set of Linear Threshold Units (neurons) that are interconnected by links, and each link has an associated connection weight. In addtion, each layer has a single Bias unit that is connected to all the units in the subsecuent layer. In theory, any three layer (input - hidden - output) MLP should be a Universal Approximator, meaning it is capable of modelling any non linear function arbitrarly well.

**- MLP Architecture**

The architecture of the MLP for this task consists of 3 layers. One input layer with 784 (28x28) nodes, followed by two hidden layers: the first with 224 nodes and the next with 288 nodes. The output layer consiste of 10 nodes representing each number from 0 to 9.

In order to make the network more robust and less prone to overfitting a dropout rate of 40% was chosen for both hidden layers. The dropout rate is a regularization method that "deactivates" a proportion of neurons, 40% in this case", to add noise to the training process and prevent co-adaptation of layers. Moreover, both hidden layers use a ReLU activation function, since it is the most common transfer fucntion because of its simplicity and ability to mitigate the vanishing gradient problem during training.

As mentioned in the beginning the output layer consists of 10 neurons, one for each number from 0 to 9. The activation function in this layer is the Softmax which assings a probability to each neuron. The neuron with the highest probability determines the model predicted number.

*Architecture Flow:*

Input (784) → Dense(224) → Dense(288) → Dense(10)

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

**Total Parameters:** 243,530  

**- Training**
- Hyperparameter Tunning
- Best model training
- Data Augmentation

<table width="100%">
  <tr>
    <th align="center">Accuracy</th>
    <th align="center">Loss</th>
  </tr>
  <tr>
    <td align="center">
      <img src="img/mlp_model_accuracy.png" width="90%" alt="MLP Accuracy Plot">
    </td>
    <td align="center">
      <img src="img/mlp_model_loss.png" width="90%" alt="MLP Loss Plot">
    </td>
  </tr>
</table>


**- Performance**
<div align="center">
  <img src="./img/mlp_model_cm.png" width="50%" alt="MLP confusion Matrix">
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

<table width="100%">
  <tr>
    <th align="center">Accuracy</th>
    <th align="center">Loss</th>
  </tr>
  <tr>
    <td align="center">
      <img src="img/cnn_model_accuracy.png" width="90%" alt="CNN Accuracy Plot">
    </td>
    <td align="center">
      <img src="img/cnn_model_loss.png" width="90%" alt="CNN Loss Plot">
    </td>
  </tr>
</table>


**- Performance**
<div align="center">
  <img src="./img/cnn_model_cm.png" width="50%" alt="CNN confusion Matrix">
</div>

---
## Performance in new domain
<div align="center">
  <img src="./img//mlp_cnn_cm.png" width="50%" alt="MLP vs. CNN Confusion Matrix">
</div>

---

## Installation & Setup

Follow the steps below to run the project locally.

### 1. Clone the Repository

```bash
git clone https://github.com/PedroGiu13/MNIST---Neural-Network.git
cd MNIST---Neural-Network
```

### 2. Create and Activate a Virtual Environment

**Windows**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Requirements

- tensorflow==2.20.0  
- keras==3.13.2  
- keras-tuner==1.4.8  
- scikit-learn==1.8.0  
- numpy==2.4.2  
- matplotlib==3.10.8  
- opencv-python==4.13.0.92  


## Usage

### Train the MLP Model

```bash
python src/MLP.py
```

### Train the CNN Model

```bash
python src/CNN.py
```

### Evaluate on Unseen USPS Data

```bash
python src/test_usps.py
```
