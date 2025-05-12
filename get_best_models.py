"""
This script processes the output of a grid search experiment (see grid_search.py),
parses all evaluation metrics, and selects the best-performing model configuration 
for each (Model, Feature) combination based on validation loss. It then outputs 
a summary table sorted by feature type and descending AUC.

Run this script after completing grid_search.py.
"""

import json
import pandas as pd
from pathlib import Path

# === Configuration ===
HERE = Path(__file__).parent
RESULTS_ROOT = HERE / 'results'

MODEL_NAMES = {
    'logistic': 'Logistic',
    'mlp': 'MLP',
    'lstm': 'LSTM',
    'cnn': 'CNN',
    'cnn_2d': 'CNN 2D',
    'wav2vec': 'Wave2Vec FT'
}

FEATURE_NAMES = {
    'mfcc': 'MFCC',
    'spectrogram_2d': 'Spectrogram',
    'wav2vec': 'Wave2Vec',
    'raw': 'Raw Waveform'
}

# === Load All Metric Files ===
metrics_paths = list(RESULTS_ROOT.rglob("metrics.json"))

# === Collect results ===
results = []

for path in metrics_paths:
    with open(path) as f:
        metrics = json.load(f)

    model_dir = path.parent
    feature_type = FEATURE_NAMES.get(model_dir.parent.name)
    
    # Load model config
    with open(model_dir / "config.json") as f:
        config = json.load(f)
    
    model_name = config.pop("model")  # remove 'model' key for get_model()
    model_name = MODEL_NAMES.get(model_name)
    
    results.append({
        "Model": model_name,
        "Features": feature_type,
        "Hyperparameters": "\n".join(str(config)[1:-1].split(', ')),
        "Accuracy": metrics.get("test_acc"),
        "F1 Score": metrics.get("f1_macro"),
        "AUC": metrics.get("auc"),
        "Val Loss": metrics.get("val_loss"),  # use this to pick best
        "Path": path.parent.relative_to(RESULTS_ROOT.parent)
    })

df = pd.DataFrame(results)

# === Keep Only Best Model Per (Model, Feature) Based on Lowest Val Loss ===
best_df = df.sort_values("Val Loss", ascending=True).drop_duplicates(["Model", "Features"])
best_df = best_df.drop(columns=["Val Loss"])

# === Sort by Feature Type then Descending AUC ===
best_df = best_df.sort_values(by=["Features", "AUC"], ascending=[True, False])

# === Save to CSV ===
best_df.to_csv(RESULTS_ROOT / 'best_models.csv', sep=',', index=False)

