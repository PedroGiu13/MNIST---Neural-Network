import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

MLP_PATH = "models/mlp_model.h5"
CNN_PATH = "models/cnn_model.h5"


def load_and_preprocess_usps():
    usps = fetch_openml("usps", version=2, as_frame=False, parser="auto")
    X_raw = usps.data
    y_raw = usps.target.astype(int)

    if y_raw.min() == 1 and y_raw.max() == 10:
        y_raw = y_raw - 1

    X_resized = []

    for row in X_raw:
        img_16 = row.reshape(16, 16)
        img_28 = cv2.resize(img_16, (28, 28), interpolation=cv2.INTER_CUBIC)
        img_28 = (img_28 - img_28.min()) / (img_28.max() - img_28.min())
        X_resized.append(img_28)

    # Return standard image shape (N, 28, 28, 1)
    X_final = np.array(X_resized).reshape(-1, 28, 28, 1)

    return X_final, y_raw


def evaluate_both_models():
    # 1. Load Data
    X_usps_img, y_usps = load_and_preprocess_usps()

    # Create Flattened version for MLP
    X_usps_flat = X_usps_img.reshape(-1, 784)

    # 2. Load Models
    try:
        mlp = tf.keras.models.load_model(MLP_PATH)
        cnn = tf.keras.models.load_model(CNN_PATH)
    except OSError as e:
        print(f"ERROR: Could not find model files. {e}")
        return

    # 3. Predict MLP (Uses Flat Data)
    pred_mlp = mlp.predict(X_usps_flat, verbose=0)
    y_pred_mlp = np.argmax(pred_mlp, axis=1)
    acc_mlp = accuracy_score(y_usps, y_pred_mlp)

    # 4. Predict CNN (Uses Image Data)
    pred_cnn = cnn.predict(X_usps_img, verbose=0)
    y_pred_cnn = np.argmax(pred_cnn, axis=1)
    acc_cnn = accuracy_score(y_usps, y_pred_cnn)

    # 5. Print Comparison
    print("\n========= USPS Benchmark Results =========")
    print(f"MLP Accuracy: {acc_mlp * 100:.2f}%")
    print(f"CNN Accuracy: {acc_cnn * 100:.2f}%")

    # 6. Side-by-Side Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Plot MLP Matrix
    cm_mlp = confusion_matrix(y_usps, y_pred_mlp)
    disp_mlp = ConfusionMatrixDisplay(confusion_matrix=cm_mlp, display_labels=range(10))
    disp_mlp.plot(cmap="Reds", ax=ax1, values_format="d", colorbar=False)
    ax1.set_title(f"MLP Confusion Matrix\nAcc: {acc_mlp:.2%}")

    # Plot CNN Matrix
    cm_cnn = confusion_matrix(y_usps, y_pred_cnn)
    disp_cnn = ConfusionMatrixDisplay(confusion_matrix=cm_cnn, display_labels=range(10))
    disp_cnn.plot(cmap="Blues", ax=ax2, values_format="d", colorbar=False)
    ax2.set_title(f"CNN Confusion Matrix\nAcc: {acc_cnn:.2%}")

    plt.suptitle("Model Robustness Comparison (USPS Dataset)", fontsize=16)

    plt.savefig("img/mlp_cnn_cm.png")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    evaluate_both_models()
