import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist

# Load .h5 file of arbitrary name for testing (last if more than one)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # path to src/
PROJECT_ROOT = os.path.dirname(BASE_DIR)  # go up one level
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

print("Looking for models in:", MODEL_DIR)

for file in os.listdir(MODEL_DIR):
    if file.endswith(".h5"):
        print("Loading:", file)
        net = load_model(os.path.join(MODEL_DIR, file))


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
print(f"ANN = {netType}")
print("Percentage correctly classified MNIST =", 100 * correct_classified / labels_test.size)
