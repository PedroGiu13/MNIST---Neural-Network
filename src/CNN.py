import keras_tuner as kt
from tensorflow import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from src.utils import plot_accuracy, plot_loss

# Hyperparameters
BATCH_SIZE = 128
EPOCHS = 30


def build_model(hp):
    input_shape = (28, 28, 1)
    num_labels = 10

    model = Sequential()

    # Input layer
    model.add(keras.Input(shape=input_shape))

    # Conv Layer 1
    hp_filters_1 = hp.Int("filters_1", min_value=32, max_value=96, step=32)
    hp_kernel = hp.Choice("kernel_size", values=[3, 5])
    model.add(Conv2D(filters=hp_filters_1, kernel_size=hp_kernel, strides=1, padding="same", activation="relu"))
    model.add(MaxPooling2D(padding="same"))

    # Check if another layer is necessary
    if hp.Boolean("use_second_layer"):
        hp_filters_2 = hp.Int("filters_2", min_value=64, max_value=256, step=64)
        model.add(Conv2D(filters=hp_filters_2, kernel_size=hp_kernel, strides=1, padding="same", activation="relu"))
        model.add(MaxPooling2D(padding="same"))

    model.add(Flatten())

    # Dense Units
    hp_dense = hp.Int("dense_units", min_value=64, max_value=256, step=64)
    model.add(Dense(units=hp_dense, activation="relu"))

    # Dropout
    hp_dropout = hp.Float("dropout", 0.0, 0.5, step=0.1)
    model.add(Dropout(hp_dropout))

    # Output Layer
    model.add(Dense(num_labels, activation="softmax"))

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
        project_name="cnn_tuning",
    )

    tuner.search(
        X_train,
        y_train,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=[keras.callbacks.EarlyStopping(patience=3)],
    )

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(f"Filters Layer 1: {best_hps.get('filters_1')}")

    if best_hps.get("use_second_layer"):
        print(f"Filters Layer 2: {best_hps.get('filters_2')}")
    else:
        print("Filters Layer 2: Not Used (Single layer model chosen)")

    print(f"Dense Units: {best_hps.get('dense_units')}")
    print(f"Dropout Rate: {best_hps.get('dropout')}")
    print(f"Learning Rate: {best_hps.get('learning_rate')}")

    best_model = tuner.hypermodel.build(best_hps)

    return best_model


def data_processing(X_raw):
    # Standardize values
    X_processed = X_raw.astype("float32") / 255.0
    # Reshape
    X_processed = X_processed.reshape((-1, 28, 28, 1))
    return X_processed


def cnn_network():
    # Import data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Process data
    X_train_full = data_processing(X_train)
    X_test = data_processing(X_test)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train, test_size=0.1, random_state=30, stratify=y_train
    )

    print("\n========= Data Information =========")
    print(f"Training Samples: {len(X_train)}")
    print(f"Validation Samples: {len(X_val)}")
    print(f"Test Samples: {len(X_test)}")

    # Tune Hyperparameters
    print("\n========= Hyperparameter Tuning =========")
    cnn = model_tuner(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, epochs=10)

    # Model Trainig
    print("\n========= Training Best Model =========")

    data_augmentation = ImageDataGenerator(
        rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1
    )

    train_data_gen = data_augmentation.flow(X_train, y_train, batch_size=BATCH_SIZE)

    history = cnn.fit(
        train_data_gen,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        # steps_per_epoch=len(X_train) // EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=1,
    )

    # Model Evaluation
    print("\n========= Best MLP Performance =========")
    test_loss, test_acc = cnn.evaluate(X_test, y_test)
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
        "Accuracy - CNN ",
        "cnn_accuracy",
    )
    plot_loss(
        history.history["loss"],
        history.history["val_loss"],
        test_loss,
        len(history.history["loss"]),
        "Loss - CNN",
        "cnn_loss",
    )

    # Save model
    cnn.save("models/cnn.h5")
    print("Model Saved")


if __name__ == "__main__":
    cnn_network()
