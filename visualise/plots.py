import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)


def plot_loss_curve(train_losses, val_losses=None, output_path=None):
    """
    Plots training (and optionally validation) loss over epochs.
    """
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    if val_losses:
        plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curve")
    
    if output_path:
        plt.savefig(output_path)
        print(f"Loss curve saved to {output_path}")
    else:
        plt.show()
    plt.close()


def plot_confusion_matrix(y_true, y_pred, labels=["Human", "Synthetic"], output_path=None):
    """
    Plots and optionally saves a confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix")

    if output_path:
        plt.savefig(output_path)
        print(f"Confusion matrix saved to {output_path}")
    else:
        plt.show()
    plt.close()


def plot_roc_curve(y_true, y_probs, output_path=None):
    """
    Plots and optionally saves the ROC curve with AUC.
    
    Args:
        y_true (list or array): Ground truth binary labels.
        y_probs (list or array): Predicted probabilities for the positive class (e.g., synthetic).
        output_path (Path or str): If provided, saves the figure to this path.
    """
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    
    if output_path:
        plt.savefig(output_path)
        print(f"ROC curve saved to {output_path}")
    else:
        plt.show()
    plt.close()

