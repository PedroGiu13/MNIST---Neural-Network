import numpy as np
import keras_tuner as kt
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from src.utils import plot_accuracy, plot_loss, plot_confusion_matrix


# Hyperparameters
BATCH_SIZE = 128
EPOCHS = 30
MODEL_NAME = "mlp_model"


def build_model(hp):
    """Build model architecture for hyperparameter tunning

    Function that defines the general structure of a CNN with the location of key structural hyperparameters that need to be tunned to find the most optima model
    Args:
        hp (keras_tunner.HyperParameters): HyperParameters object used to define the search space for the model

    Returns:
        model: compiled keras model with initial hyperparameters, ready for training
    """
    # Define shape of inputs (features) and number of expected outputs (labels)
    input_size = (784,)
    num_labels = 10

    # Initialize model selection
    model = Sequential()

    # Input Layer
    model.add(keras.Input(shape=input_size))

    # Hidden Layer 1:
    # HP to tune: number of units (32 - 512), dropout rate (0.0 - 0.5)
    hp_units_1 = hp.Int("units_1", min_value=32, max_value=512, step=32)
    model.add(Dense(units=hp_units_1))
    model.add(Activation("relu"))
    hp_dropout = hp.Float("dropout", min_value=0.0, max_value=0.5, step=0.1)
    model.add(Dropout(hp_dropout))

    # Hidden Layer 2:
    # HP to tune: number of units (32 - 512)
    # Same Dropout
    hp_units_2 = hp.Int("units_2", min_value=32, max_value=512, step=32)
    model.add(Dense(hp_units_2))
    model.add(Activation("relu"))
    model.add(Dropout(hp_dropout))

    # Output Layer
    model.add(Dense(num_labels))
    model.add(Activation("softmax"))

    # Learning Rate: hyperparameter = learning rate
    hp_eta = hp.Choice("learning_rate", values=[0.01, 0.001, 0.0001])

    # Compile entire model
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=hp_eta),
        metrics=["accuracy"],
    )

    return model


def model_tuner(X_train, y_train, X_val, y_val, epochs):
    """Tune hyperparameters for most optimal CNN

    This funtion iterates through all the hyperparameters defined in the model. Tests the performance on the validation set (accuracy) and chooses the parameters that produce the highest accuracy as the best model

    Args:
        X_train np.array: training features
        y_train np.array: training labels
        X_val np.array: validation features
        y_val np.array: validation labels
        epochs int: number od iterations

    Returns:
        best_model: model with the best hyperparameters
    """
    # Initialize Random Search with the given model structure and other parameters
    tuner = kt.RandomSearch(
        build_model,
        objective="val_accuracy",
        max_trials=10,
        executions_per_trial=1,
        directory="hyperparameter_tuning",
        project_name="mlp_tuning",
    )

    # Iterative process to find the most optimal model
    tuner.search(
        X_train,
        y_train,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=[keras.callbacks.EarlyStopping(patience=3)],
    )

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(f"Units Layer 1: {best_hps.get('units_1')}")
    print(f"Units Layer 2: {best_hps.get('units_2')}")
    print(f"Dropout Rate: {best_hps.get('dropout')}")
    print(f"Learning Rate: {best_hps.get('learning_rate')}")

    best_model = tuner.hypermodel.build(best_hps)

    return best_model


def data_processing(X_raw):
    """Process data to meet model input requirements

    Args:
        X_raw (np.array): raw dataset of features

    Returns:
        _X_processed: processed data
    """
    # Standardize values
    X_processed = X_raw.astype("float32") / 255.0
    # Reshape
    X_processed = X_processed.reshape((-1, 784))

    return X_processed


def mlp_network():
    """Orchestrator function.

    This function is responsible taking all the steps required to import, build, train, predict, and plot outputs of the model
    """
    # Import data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Process data
    X_train = data_processing(X_train)
    X_test = data_processing(X_test)

    X_train_hp, X_val_hp, y_train_hp, y_val_hp = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    # Tune Hyperparameters
    print("\n========= Hyperparameter Tuning =========")
    mlp = model_tuner(X_train=X_train_hp, y_train=y_train_hp, X_val=X_val_hp, y_val=y_val_hp, epochs=10)

    # Model Trainig
    print("\n========= Training Best Model =========")
    history = mlp.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=1,
    )

    # Model Prediction
    mlp_pred = mlp.predict(X_test)
    y_pred = np.argmax(mlp_pred, axis=1)

    # Model Evaluation
    print("\n========= Best MLP Performance =========")
    test_loss, test_acc = mlp.evaluate(X_test, y_test)
    print(f"Train Accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    print(f"Train Loss: {history.history['loss'][-1]:.4f}")
    print(f"Validation Loss: {history.history['val_loss'][-1]:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

    # Visualize the model
    plot_accuracy(
        history.history["accuracy"],
        history.history["val_accuracy"],
        test_acc,
        len(history.history["accuracy"]),
        "Accuracy - MLP ",
        f"{MODEL_NAME}_accuracy",
    )
    plot_loss(
        history.history["loss"],
        history.history["val_loss"],
        test_loss,
        len(history.history["loss"]),
        "Loss - MLP",
        f"{MODEL_NAME}_loss",
    )

    plot_confusion_matrix(
        y_mnist=y_test, y_pred=y_pred, title=f"{MODEL_NAME}- Confusion Matrix", file_name=f"{MODEL_NAME}_cm"
    )

    # Save model
    mlp.save(f"models/{MODEL_NAME}.h5")
    print("Model Saved")


if __name__ == "__main__":
    mlp_network()
