"""Module to plot accuracy, loss and confusion matrix in neural networks"""

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plot_accuracy(train_acc, val_acc, test_acc, epochs, title, file_name):
    """Plot accuracy evolution (training/validation) during model training

    This function plots the accuracy results, of both the training and validation sets, of the model during each epoch. In the end include a line with the final test accuracy for comparison. The plot is saved and displayed.

    Args:
        train_acc (list[float]): accuracy in training dataset
        val_acc (list[float]): accuracy in validation dataset
        test_acc (float): accuracy in testing dataset
        epochs (int): number of epochs during training
        title (str): title of the chart
        file_name (str): name of file for saving
    """
    epochs_range = range(1, epochs + 1)

    plt.figure(figsize=(14, 6))
    plt.suptitle(title, fontsize=16)

    plt.subplot()
    plt.plot(epochs_range, train_acc, color="blue", label="Train Accuracy")
    plt.plot(epochs_range, val_acc, color="red", label="Validation Accuracy")
    plt.axhline(y=test_acc, color="green", linestyle="--", linewidth=2, label=f"Final Test Score ({test_acc:.4f})")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save img
    save_path = f"img/{file_name}.png"
    plt.savefig(save_path)

    plt.show()


def plot_loss(train_loss, val_loss, test_loss, epochs, title, file_name):
    """Plot loss evolution (training/validation) during model training

    This function plots the loss results, of both the training and validation sets, of the model during each epoch. In the end include a line with the final test loss for comparison. The plot is saved and displayed.

    Args:
        train_acc (list[float]): loss in training dataset
        val_acc (list[float]): loss in validation dataset
        test_acc (float): loss in testing dataset
        epochs (int): number of epochs during training
        title (str): title of the chart
        file_name (str): name of file for saving
    """
    epochs_range = range(1, epochs + 1)

    plt.figure(figsize=(14, 6))
    plt.suptitle(title, fontsize=16)

    plt.subplot()
    plt.plot(epochs_range, train_loss, color="blue", label="Train Loss")
    plt.plot(epochs_range, val_loss, color="red", label="Validation Loss")
    plt.axhline(y=test_loss, color="green", linestyle="--", linewidth=2, label=f"Final Test Score ({test_loss:.4f})")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save img
    save_path = f"img/{file_name}.png"
    plt.savefig(save_path)

    plt.show()


def plot_confusion_matrix(y_mnist, y_pred, title, file_name):
    """Creates confusion matrix to compare actual labels with predicted labels

    This function creates a confusion matrix using 'sklearn.metrics.ConfusionMatrixDisplay' to compare in which numbers did the network struggled the most and vice versa. The function saves the confusion matrix and displays it.

    Args:
        y_mnist (list[int]): list of real labels
        y_pred (list[int]): list of prediction of trained model
        title (str): chart title
        file_name (str): file name for saving
    """
    cm = confusion_matrix(y_mnist, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))

    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(cmap="Blues", ax=ax, values_format="d")
    plt.title(title, fontsize=16)
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)

    # Save img
    save_path = f"img/{file_name}.png"
    plt.savefig(save_path, bbox_inches="tight")

    plt.show()
