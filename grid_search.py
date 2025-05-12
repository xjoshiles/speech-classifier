from itertools import product
import copy
import json
from pathlib import Path
import torch
from torch import nn
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

from dataset import load_dataloaders
from models import get_model
from train import train_model
from evaluate import evaluate_model
from config import USE_GPU
from visualise.plots import plot_loss_curve, plot_confusion_matrix, plot_roc_curve

SUPPORTED_FEATURES = ["mfcc", "spectrogram_2d", "wav2vec", "raw"]


def save_experiment_output(
        output_dir,
        model,
        config,
        metrics,
        loss_history=None,
        val_history=None,
        y_true=None,
        y_pred=None,
        y_probs=None
        ):
    """
    Saves model checkpoint, config, metrics, and optionally:
    - Loss curves
    - Confusion matrix
    - Classification report
    - ROC curve
    - Additional derived metrics
    
    Args:
        output_dir (Path): Directory to save outputs to.
        model (nn.Module): Trained PyTorch model.
        config (dict): Model configuration and hyperparameters.
        metrics (dict): Dictionary of evaluation metrics.
        loss_history (list[float], optional): Training loss values.
        val_history (list[float], optional): Validation loss values.
        y_true (list[int], optional): True labels.
        y_pred (list[int], optional): Predicted labels.
        y_probs (list[float], optional): Predicted probabilities for class 1.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model weights and config
    torch.save(model.state_dict(), output_dir / "model.pt")
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Save loss curves
    if loss_history and val_history:
        plot_loss_curve(loss_history, val_history, output_path=output_dir / "loss_curve.png")
    
    if y_true is not None and y_pred is not None:
        # Save confusion matrix
        plot_confusion_matrix(y_true, y_pred, output_path=output_dir / "confusion_matrix.png")
        
        # Save classification report
        report = classification_report(
            y_true, y_pred, target_names=["Human", "Synthetic"], digits=4
        )
        with open(output_dir / "classification_report.txt", "w") as f:
            f.write(report)
        
        # Extract additional metrics from classification report
        report_dict = classification_report(
            y_true, y_pred, target_names=["Human", "Synthetic"], output_dict=True
        )
        metrics.update({
            "f1_macro": round(report_dict["macro avg"]["f1-score"], 4),
            "f1_weighted": round(report_dict["weighted avg"]["f1-score"], 4),
            "precision_human": round(report_dict["Human"]["precision"], 4),
            "recall_human": round(report_dict["Human"]["recall"], 4),
            "precision_synth": round(report_dict["Synthetic"]["precision"], 4),
            "recall_synth": round(report_dict["Synthetic"]["recall"], 4),
        })
    
    # Save ROC curve
    if y_true is not None and y_probs is not None:
        plot_roc_curve(y_true, y_probs, output_path=output_dir / "roc_curve.png")
    
    # Save metrics
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


def get_loss_fn_from_loader(loader,
                            device,
                            label="",
                            skip_if_balanced=True,
                            tolerance=0.01):
    """
    Returns a CrossEntropyLoss function, optionally weighted to correct class imbalance.

    Uses class-balanced weights from the dataset if available. If the class weights
    are approximately equal (i.e., the dataset is balanced), uses unweighted loss.

    Args:
        loader (DataLoader): A PyTorch DataLoader with a dataset that optionally provides `compute_class_weights()`.
        device (torch.device): The device to put weights on.
        label (str): Optional label for logging (e.g., 'train', 'val', 'test').
        skip_if_balanced (bool): Whether to skip weighting if weights are nearly equal.
        tolerance (float): Acceptable relative difference between class weights for skipping weighting.

    Returns:
        nn.CrossEntropyLoss: A weighted or unweighted loss function.
    """
    if hasattr(loader.dataset, "compute_class_weights"):
        class_weights = loader.dataset.compute_class_weights().to(device)

        if skip_if_balanced:
            rel_diff = torch.abs(class_weights[0] - class_weights[1]) / class_weights.mean()
            if rel_diff < tolerance:
                print(f"[INFO] {label.capitalize()} labels are balanced (Δrel={rel_diff:.2%} < tol={tolerance:.2%}). Using unweighted loss.")
                return nn.CrossEntropyLoss()

        print(f"[INFO] Using weighted loss for {label}: class 0 = {class_weights[0]:.4f}, class 1 = {class_weights[1]:.4f}")
        return nn.CrossEntropyLoss(weight=class_weights)

    print(f"[INFO] {label.capitalize()} dataset has no weighting method. Using unweighted loss.")
    return nn.CrossEntropyLoss()


def train_all_models(model_configs,
                     train_loader,
                     val_loader,
                     test_loader,
                     feature_type="unknown",
                     use_weighted_loss=True
                     ):
    """
    Trains and evaluates all model configurations for a given feature type and saves results to disk.
    
    This function:
    - Computes class-weighted loss if enabled.
    - Iterates over all hyperparameter combinations for each model type.
    - Skips training if model artifacts already exist.
    - Evaluates each trained model on validation and test sets.
    - Saves training curves, metrics, predictions, and config for each experiment.
    
    Args:
        model_configs (dict): Dictionary of model names and hyperparameter grids.
        train_loader (DataLoader): Training set DataLoader.
        val_loader (DataLoader): Validation set DataLoader.
        test_loader (DataLoader): Test set DataLoader.
        feature_type (str): The input feature type (e.g., 'mfcc', 'wav2vec').
        use_weighted_loss (bool): Whether to apply class-balanced loss during training.
    """
    device = torch.device("cuda" if USE_GPU else "cpu")
    
    # ===== Compute loss functions =====
    if use_weighted_loss:
        print("\nComputing class-balanced weights for loss functions...")
        train_loss_fn = get_loss_fn_from_loader(train_loader, device, label="train")
        val_loss_fn   = get_loss_fn_from_loader(val_loader, device, label="val")
        test_loss_fn  = get_loss_fn_from_loader(test_loader, device, label="test")
    else:
        train_loss_fn = val_loss_fn = test_loss_fn = nn.CrossEntropyLoss()
    
    
    for model_name, param_grid in model_configs.items():
        keys, values = zip(*param_grid.items())
        for param_values in product(*values):
            params = dict(zip(keys, param_values))
            
            experiment_name = f"{model_name}_" + "_".join(f"{k}{v}" for k, v in params.items())
            output_dir = Path("results") / feature_type / experiment_name
            
            # Skip already trained models
            model_path = output_dir / "model.pt"
            config_path = output_dir / "config.json"
            if model_path.exists() and config_path.exists():
                print(f"⏩ Skipping {experiment_name} (already completed)")
                continue
            
            print(f"🧪 Training {model_name.upper()} with {params}")
            
            # Determine input_dim from a sample batch
            for batch in train_loader:
                X = batch["input"]
                if "cnn_2d" in model_name:
                    input_dim = X.shape[1]   # Channel dimension
                elif "wav2vec" in model_name:
                    input_dim = None         # Wav2Vec2 handles this internally
                else:
                    input_dim = X.shape[-1]  # Feature dimension for 1D models
                break
            
            # Load and train the model
            model = get_model(model_name, input_dim=input_dim, **params)
            model, train_loss_history, val_loss_history = train_model(
                model,
                train_loader,
                val_loader,
                loss_fn=train_loss_fn,
                val_loss_fn=val_loss_fn,
                return_loss_history=True
            )
            
            # === Evaluate on validation set ===
            val_loss, val_acc = evaluate_model(
                model, val_loader, device, loss_fn=val_loss_fn)
            print(f"✅ {model_name.upper()} Val Acc: {val_acc:.2%} | Val Loss: {val_loss:.4f}")
            
            # === Evaluate on test set ===
            test_loss, test_acc, y_test, y_pred, y_probs = evaluate_model(
                model,
                test_loader,
                device,
                return_preds=True,
                return_probs=True,
                loss_fn=test_loss_fn
            )
            print(f"🧪 {model_name.upper()} Test Acc: {test_acc:.2%} | Test Loss: {test_loss:.4f}\n")
            
            # === Save experiment ===
            metrics = {
                "val_loss": val_loss,
                "val_acc": val_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "early_stopping_epoch": len(train_loss_history),
                "auc": round(roc_auc_score(y_test, y_probs), 4),
            }
            
            save_experiment_output(
                output_dir=output_dir,
                model=model,
                config={"model": model_name, **params},
                metrics=metrics,
                loss_history=train_loss_history,
                val_history=val_loss_history,
                y_true=y_test,
                y_pred=y_pred,
                y_probs=y_probs
            )


def main():
    for feature_type in SUPPORTED_FEATURES:
        print(f"\n=== Training models for feature: {feature_type} ===")
        
        # Load data loaders for this feature type
        train_loader, val_loader, test_loader = load_dataloaders(
            split_mode="tts_system",
            feature_type=feature_type
            )
        
        # Automatically determine the number of domains in the training set
        num_domains = int(train_loader.dataset.domains.max().item()) + 1
        
        if feature_type == "spectrogram_2d":
            # Only include 2D CNN models for spectrogram_2d
            model_configs = {
                "cnn_2d": {
                    "num_filters": [64, 128],
                    "kernel_size": [3, 5],
                    "num_conv_layers": [1, 2],
                    "dropout": [0.0, 0.3],
                    "pooling": ["adaptive", "max"],
                    "activation_fn": ["relu", "gelu"],
                    "hidden_dim": [128],
                }
            }
        
        elif feature_type == "raw":
            # Only include Wav2Vec fine-tuning models for raw audio
            model_configs = {
                "wav2vec": {
                    "hidden_dim": [128],
                    "dropout": [0.0, 0.1, 0.3],
                    "freeze_feature_extractor": [True, False],
                }
            }
        
        else:
            # Set base configs for 1D-compatible non-raw features (MFCC, Wav2vec)
            model_configs = {
                "lstm": {
                    "hidden_dim": [64, 128],
                    "num_layers": [1, 2],
                    "bidirectional": [True, False],
                    "dropout": [0.0, 0.1, 0.3],
                },
                "cnn": {
                    "num_filters": [64, 128],
                    "kernel_size": [3, 5],
                    "num_conv_layers": [1, 2],
                    "dropout": [0.0, 0.3],
                    "pooling": ["adaptive", "max"],
                    "activation_fn": ["relu", "gelu"],
                    "hidden_dim": [128],
                },
                "logistic": {
                    "use_bias": [True, False],
                    "dropout": [0.0, 0.3],
                },
                "mlp": {
                    "hidden_dims": [[128], [256, 128]],
                    "dropout": [0.0, 0.2, 0.5],
                    "activation_fn": ["relu", "gelu"],
                }
            }
        
        train_all_models(
            model_configs,
            train_loader,
            val_loader,
            test_loader,
            feature_type=feature_type,
            use_weighted_loss=True
            )
    
    print('Training complete for all features')


if __name__ == "__main__":
    main()
