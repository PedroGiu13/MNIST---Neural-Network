import keras_tuner as kt
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from src.utils import plot_accuracy, plot_loss


# Hyperparameters
BATCH_SIZE = 128
EPOCHS = 30


def build_model(hp):
    input_size = (784,)
    num_labels = 10

    model = Sequential()

    # Input Layer
    model.add(keras.Input(shape=input_size))

    # Hidden Layer 1
    hp_units_1 = hp.Int("units_1", min_value=32, max_value=512, step=32)
    model.add(Dense(units=hp_units_1))
    model.add(Activation("relu"))

    hp_dropout = hp.Float("dropout", min_value=0.0, max_value=0.5, step=0.1)
    model.add(Dropout(hp_dropout))

    # Hidden Layer 2
    hp_units_2 = hp.Int("units_2", min_value=32, max_value=512, step=32)
    model.add(Dense(hp_units_2))
    model.add(Activation("relu"))
    model.add(Dropout(hp_dropout))

    # Output Layer
    model.add(Dense(num_labels))
    model.add(Activation("softmax"))

    # Model Training
    hp_eta = hp.Choice("learning_rate", values=[0.01, 0.001, 0.0001])
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=hp_eta),
        metrics=["accuracy"],
    )

    return model


def model_tuner(X_train, y_train, X_val, y_val, epochs):
    tuner = kt.RandomSearch(
        build_model,
        objective="val_accuracy",
        max_trials=10,
        executions_per_trial=1,
        directory="hyperparameter_tuning",
        project_name="mlp_tuning",
    )

    tuner.search(
        X_train,
        y_train,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=[keras.callbacks.EarlyStopping(patience=3)],
    )

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print("\n===== Best Hyperparameters Found =====")
    print(f"Units Layer 1: {best_hps.get('units_1')}")
    print(f"Units Layer 2: {best_hps.get('units_2')}")
    print(f"Dropout Rate: {best_hps.get('dropout')}")
    print(f"Learning Rate: {best_hps.get('learning_rate')}")

    best_model = tuner.hypermodel.build(best_hps)

    return best_model


def data_processing(X_raw):
    # Standardize values
    X_processed = X_raw.astype("float32") / 255.0
    # Reshape
    X_processed = X_processed.reshape((-1, 784))

    return X_processed


def mlp_network():
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
    history = mlp.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1, verbose=1)

    # Model Evaluation
    print("\n========= Best MLP Performance =========")
    test_loss, test_acc = mlp.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

    # Visualize the model
    plot_accuracy(
        history.history["accuracy"],
        history.history["val_accuracy"],
        test_acc,
        EPOCHS,
        "Accuracy - MLP ",
        "mlp_accuracy",
    )
    plot_loss(history.history["loss"], history.history["val_loss"], test_loss, EPOCHS, "Loss - MLP", "mlp_loss")

    # Save model
    mlp.save("/models/mlp.h5")
    print("Model Saved")


if __name__ == "__main__":
    mlp_network()
