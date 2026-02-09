import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist

# Load .h5 file of arbitrary name for testing (last if more than one)
print(os.getcwd())
for file in os.listdir(os.getcwd()):
    if file.endswith(".h5"):
        print(file)
        net = load_model(file)
net.summary()


# Determine what type of network this is
input_dims = net.input_shape
netType = "CNN" if len(input_dims) > 2 else "MLP"

# Test with MNIST data
(x_train, labels_train), (x_test, labels_test) = mnist.load_data()
x_test = x_test.astype("float32") / 255

if netType == "MLP":
    x_test = x_test.reshape(10000, 784)
else:
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Evaluate
outputs = net.predict(x_test)
labels_predicted = np.argmax(outputs, axis=1)
correct_classified = np.sum(labels_predicted == labels_test)
print("Percentage correctly classified MNIST =", 100 * correct_classified / labels_test.size)
