"""
classical_model.py
==================
RandomForest-based binder classifier.

Design choices
--------------
* class_weight='balanced'  — automatically up-weights the minority class
  (binders) in proportion to their frequency, so the model does not simply
  predict "non-binder" for everything.
* Threshold tuning  — after training, we search for the lowest decision
  threshold that still achieves ≥85 % binder recall on a validation split,
  trading some precision for maximum hit-rate coverage.
* Full 2048-dim Morgan fingerprints are used here (not PCA) because
  RandomForest handles high-dimensional binary features well without scaling.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for safe saving
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

DEFAULT_N_ESTIMATORS: int = 300
DEFAULT_MIN_RECALL:   float = 0.85  # target recall when tuning threshold


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_classical_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    random_state: int = 42,
) -> RandomForestClassifier:
    """
    Train a balanced RandomForest on Morgan fingerprints.

    class_weight='balanced' sets  w_class = n_samples / (n_classes * n_class_i)
    so that each tree effectively sees equal representation of both classes
    without discarding data via undersampling.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        max_features="sqrt",
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=random_state,
        n_jobs=-1,
        verbose=0,
    )
    model.fit(X_train, y_train)
    logger.info(
        f"RandomForest trained  |  "
        f"{n_estimators} trees  |  {X_train.shape[0]} samples"
    )
    return model


def tune_threshold_for_recall(
    model: RandomForestClassifier,
    X_val: np.ndarray,
    y_val: np.ndarray,
    min_recall: float = DEFAULT_MIN_RECALL,
) -> float:
    """
    Sweep decision thresholds from 0.10 → 0.80 and return the first threshold
    that achieves ≥ min_recall on the validation set.

    A lower threshold means more compounds are flagged as binders — better
    recall but lower precision.  The hybrid pipeline's QML stage can then
    re-rank and filter false positives.
    """
    probas = model.predict_proba(X_val)[:, 1]
    best_threshold = 0.5   # sensible default

    for thresh in np.arange(0.45, 0.05, -0.02):
        preds = (probas >= thresh).astype(int)
        rec   = recall_score(y_val, preds, zero_division=0)
        if rec >= min_recall:
            best_threshold = round(float(thresh), 2)
            break

    logger.info(
        f"Threshold tuning: {best_threshold:.2f} → "
        f"recall≥{min_recall} on validation set"
    )
    return best_threshold


# ─────────────────────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────────────────────

def predict_with_proba(
    model: RandomForestClassifier,
    X: np.ndarray,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (binary_predictions, binder_probabilities).
    Probabilities are the soft score used by the hybrid pipeline for ranking.
    """
    probas = model.predict_proba(X)[:, 1]
    preds  = (probas >= threshold).astype(int)
    return preds, probas


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_classical_model(
    model: RandomForestClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5,
    save_plots: bool = True,
) -> Dict[str, float]:
    """
    Compute and print accuracy, precision, recall, F1, ROC-AUC.
    Optionally save a confusion-matrix heatmap.
    """
    preds, probas = predict_with_proba(model, X_test, threshold)

    metrics = {
        "accuracy":  accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall":    recall_score(y_test, preds, zero_division=0),
        "f1":        f1_score(y_test, preds, zero_division=0),
        "roc_auc":   roc_auc_score(y_test, probas),
    }

    print("\n" + "═" * 60)
    print("  Classical Model (RandomForest) — Evaluation Report")
    print("═" * 60)
    print(f"  Threshold : {threshold}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}  ← maximised")
    print(f"  F1        : {metrics['f1']:.4f}")
    print(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")
    print()
    print(
        classification_report(
            y_test, preds, target_names=["Non-binder", "Binder"]
        )
    )

    if save_plots:
        _save_confusion_matrix_plot(
            confusion_matrix(y_test, preds),
            title="Classical RF — Confusion Matrix",
            filepath=MODELS_DIR / "classical_confusion_matrix.png",
        )

    return metrics


def _save_confusion_matrix_plot(
    cm: np.ndarray,
    title: str,
    filepath: Path,
) -> None:
    """Save a labelled confusion-matrix heatmap as PNG."""
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Non-binder", "Binder"],
        yticklabels=["Non-binder", "Binder"],
        ax=ax,
    )
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    logger.info(f"Confusion matrix saved → {filepath}")


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────

def save_classical_model(
    model: RandomForestClassifier,
    threshold: float,
    path: Path = MODELS_DIR / "classical_model.pkl",
) -> None:
    """Persist model and its tuned decision threshold together."""
    joblib.dump({"model": model, "threshold": threshold}, path, compress=3)
    logger.info(f"Classical model saved → {path}")


def load_classical_model(
    path: Path = MODELS_DIR / "classical_model.pkl",
) -> Tuple[RandomForestClassifier, float]:
    """Load a previously saved RandomForest + threshold pair."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Classical model not found at {path}. "
            "Run `python classical_model.py` or `python train.py` first."
        )
    bundle = joblib.load(path)
    return bundle["model"], bundle["threshold"]


# ─────────────────────────────────────────────────────────────────────────────
# Convenience wrapper called by train.py
# ─────────────────────────────────────────────────────────────────────────────

def train_and_save_classical_model(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.20,
    val_size:  float = 0.15,
) -> Tuple[RandomForestClassifier, float, Dict[str, float]]:
    """
    Full training routine:
      1. Stratified train / val / test split
      2. Train RandomForest on the training portion
      3. Tune threshold on the validation set
      4. Evaluate on the held-out test set
      5. Save model + threshold to disk

    Returns (model, threshold, metrics_dict).
    """
    # Outer train/test split
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # Inner train/val split for threshold tuning
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=val_size,
        random_state=42,
        stratify=y_train_full,
    )

    model     = train_classical_model(X_train, y_train)
    threshold = tune_threshold_for_recall(model, X_val, y_val)
    metrics   = evaluate_classical_model(model, X_test, y_test, threshold)
    save_classical_model(model, threshold)

    return model, threshold, metrics


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from preprocessing import load_processed_data

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    data  = load_processed_data()
    model, threshold, metrics = train_and_save_classical_model(
        data["X_fp"], data["y"]  # full 2048-dim fingerprints for RF
    )
    print(f"\nBest threshold: {threshold}")
    print(f"Recall on test set: {metrics['recall']:.4f}")
