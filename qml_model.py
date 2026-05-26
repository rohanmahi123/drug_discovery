"""
qml_model.py
============
Variational Quantum Classifier (VQC) for drug-binding prediction.

Architecture
------------
  Feature map : ZZFeatureMap (encodes classical data into quantum phases)
  Ansatz      : RealAmplitudes (parameterised rotation + CNOT entanglement)
  Optimiser   : SPSA — Simultaneous Perturbation Stochastic Approximation.
                SPSA is gradient-free and well-suited for noisy or simulated
                quantum circuits where analytic gradients are expensive.

Input dimensionality
--------------------
8-dimensional PCA-reduced features are used so the circuit requires only
8 qubits — staying within practical simulation budgets.  Features are
rescaled to [0, π] before encoding, matching the ZZFeatureMap phase range.

Training budget
---------------
QML simulation is exponentially costly.  QML_SUBSAMPLE (default 500) caps
the training set with class balancing so both binder/non-binder classes
contribute equally.  Increase SPSA_MAXITER for better convergence at the
cost of runtime.

Disclaimer
----------
No quantum advantage is claimed.  This VQC runs on a classical statevector
simulator.  The architecture is designed for correctness and reproducibility,
not performance superiority over classical methods.
"""

import logging
import pickle
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

QML_SUBSAMPLE: int  = 500    # max training compounds (keeps simulation tractable)
NUM_FEATURES:  int  = 8      # PCA output dimension == number of qubits
VQC_REPS:      int  = 2      # repetitions for feature map AND ansatz layers
SPSA_MAXITER:  int  = 100    # SPSA optimiser iterations


# ─────────────────────────────────────────────────────────────────────────────
# Qiskit import helper  (handles API differences across qiskit versions)
# ─────────────────────────────────────────────────────────────────────────────

def _import_qiskit_components():
    """
    Try several import paths to stay compatible with qiskit 1.x + qiskit-ml 0.7.x.
    Returns (ZZFeatureMap, RealAmplitudes, VQC, SPSA, Sampler_or_None).
    """
    from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes  # stable since 0.x

    # VQC
    try:
        from qiskit_machine_learning.algorithms import VQC
    except ImportError as exc:
        raise ImportError(
            "qiskit-machine-learning is required.  "
            "Install with: pip install qiskit-machine-learning"
        ) from exc

    # SPSA optimiser
    try:
        from qiskit_algorithms.optimizers import SPSA
    except ImportError:
        try:
            from qiskit.algorithms.optimizers import SPSA
        except ImportError as exc:
            raise ImportError(
                "SPSA optimiser not found.  "
                "Install qiskit-algorithms: pip install qiskit-algorithms"
            ) from exc

    # Sampler primitive (required by qiskit-ml 0.7+)
    sampler = None
    try:
        from qiskit.primitives import StatevectorSampler
        sampler = StatevectorSampler()
    except ImportError:
        try:
            from qiskit.primitives import Sampler
            sampler = Sampler()
        except ImportError:
            pass   # older qiskit-ml builds the sampler internally

    return ZZFeatureMap, RealAmplitudes, VQC, SPSA, sampler


# ─────────────────────────────────────────────────────────────────────────────
# VQC Construction
# ─────────────────────────────────────────────────────────────────────────────

def build_vqc(
    num_features: int = NUM_FEATURES,
    reps: int = VQC_REPS,
    max_iter: int = SPSA_MAXITER,
) -> object:
    """
    Build a VQC instance.

    ZZFeatureMap
        Maps each feature x_i to a phase rotation, then applies ZZ-type
        entangling gates between pairs of qubits. reps=2 gives one layer of
        single-qubit encoding followed by one layer of two-qubit entanglement.

    RealAmplitudes
        Alternates layers of Ry rotations (trainable parameters) with CNOT
        chains (entanglement='linear').  Only real amplitudes are used —
        equivalent to using no global phase, which is sufficient for
        classification.

    SPSA
        Estimates the gradient by perturbing all parameters simultaneously
        by ±δ.  Robust to noise and avoids the barren-plateau problem that
        affects gradient-based methods at scale.
    """
    ZZFeatureMap, RealAmplitudes, VQC, SPSA, sampler = _import_qiskit_components()

    feature_map = ZZFeatureMap(feature_dimension=num_features, reps=reps)
    ansatz      = RealAmplitudes(
        num_qubits=num_features, reps=reps, entanglement="linear"
    )
    optimizer   = SPSA(maxiter=max_iter)

    # Build VQC — handle sampler argument gracefully
    try:
        if sampler is not None:
            vqc = VQC(
                feature_map=feature_map,
                ansatz=ansatz,
                optimizer=optimizer,
                sampler=sampler,
            )
        else:
            vqc = VQC(
                feature_map=feature_map,
                ansatz=ansatz,
                optimizer=optimizer,
            )
    except TypeError:
        # Some versions do not accept sampler as constructor arg
        vqc = VQC(
            feature_map=feature_map,
            ansatz=ansatz,
            optimizer=optimizer,
        )

    logger.info(
        f"VQC built  |  {num_features} qubits  |  {reps} reps  |  "
        f"SPSA({max_iter} iters)"
    )
    return vqc


# ─────────────────────────────────────────────────────────────────────────────
# Feature Preprocessing for Quantum Encoding
# ─────────────────────────────────────────────────────────────────────────────

def rescale_to_pi(X: np.ndarray) -> np.ndarray:
    """
    Linearly rescale each feature column to [0, π].

    ZZFeatureMap uses phase rotations of the form exp(i x_i x_j), so keeping
    features in [0, π] prevents aliasing from wrapped phases.
    """
    X_min = X.min(axis=0, keepdims=True)
    X_max = X.max(axis=0, keepdims=True)
    denom = np.where(X_max - X_min == 0, 1.0, X_max - X_min)
    return (X - X_min) / denom * np.pi


# ─────────────────────────────────────────────────────────────────────────────
# Balanced Subsampling
# ─────────────────────────────────────────────────────────────────────────────

def subsample_balanced(
    X: np.ndarray,
    y: np.ndarray,
    n: int = QML_SUBSAMPLE,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Draw at most n/2 binders and n/2 non-binders to keep the training set
    balanced and within the quantum simulation budget.
    """
    rng   = np.random.default_rng(random_state)
    idx_1 = np.where(y == 1)[0]
    idx_0 = np.where(y == 0)[0]
    half  = min(n // 2, len(idx_1), len(idx_0))

    if half == 0:
        logger.warning("Subsample: one class is empty — returning full dataset.")
        return X, y

    sel_1 = rng.choice(idx_1, half, replace=False)
    sel_0 = rng.choice(idx_0, half, replace=False)
    idx   = np.concatenate([sel_1, sel_0])
    rng.shuffle(idx)

    logger.info(
        f"QML subsample: {len(idx)} compounds "
        f"({half} binders + {half} non-binders)"
    )
    return X[idx], y[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_qml_model(
    X_pca: np.ndarray,
    y: np.ndarray,
    max_iter: int = SPSA_MAXITER,
    subsample: int = QML_SUBSAMPLE,
) -> object:
    """
    Train VQC on balanced subsample of 8-dim PCA features.

    Training can take 10–60 minutes on CPU depending on SPSA_MAXITER and
    subsample size.  Reduce max_iter to 30–50 for quick experiments.
    """
    X_sub, y_sub = subsample_balanced(X_pca, y, n=subsample)
    X_sub        = rescale_to_pi(X_sub)

    logger.info(
        f"Starting QML training  |  {len(y_sub)} samples  |  "
        f"SPSA maxiter={max_iter}  |  This may take several minutes …"
    )
    vqc = build_vqc(
        num_features=X_sub.shape[1], reps=VQC_REPS, max_iter=max_iter
    )
    vqc.fit(X_sub, y_sub)
    logger.info("VQC training complete.")
    return vqc


# ─────────────────────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────────────────────

def predict_qml(
    vqc,
    X_pca: np.ndarray,
    apply_rescale: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run VQC inference and return (binary_labels, confidence_scores).

    Qiskit-ml 0.7+ VQC.predict() may return one-hot 2-D arrays like
    [[0, 1], [1, 0], ...] rather than flat [1, 0, ...].
    We normalise both cases to a plain 1-D integer array.
    """
    X_in = rescale_to_pi(X_pca) if apply_rescale else X_pca.copy()

    raw = vqc.predict(X_in)

    # Convert one-hot → binary label
    if raw.ndim == 2:
        preds = np.argmax(raw, axis=1).astype(int)
    else:
        preds = raw.astype(int)

    # Confidence scores: use predict_proba when available
    try:
        proba_2d = vqc.predict_proba(X_in)
        if proba_2d.ndim == 2:
            probas = proba_2d[:, 1]          # P(binder)
        else:
            probas = proba_2d.astype(float)
    except (AttributeError, Exception):
        # Fall back to using the argmax distance as a soft score
        if raw.ndim == 2:
            total = raw.sum(axis=1, keepdims=True)
            total = np.where(total == 0, 1, total)
            probas = (raw / total)[:, 1].astype(float)
        else:
            probas = preds.astype(float)

    return preds, probas


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_qml_model(
    vqc,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """
    Compute accuracy, precision, recall, F1 for the QML model.

    Note: QML metrics on a classical simulation should NOT be used to
    claim quantum advantage.  They serve as a sanity check that the VQC
    has learned something meaningful.
    """
    preds, _ = predict_qml(vqc, X_test)

    metrics = {
        "accuracy":  accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall":    recall_score(y_test, preds, zero_division=0),
        "f1":        f1_score(y_test, preds, zero_division=0),
    }

    print("\n" + "═" * 60)
    print("  Quantum ML Model (VQC) — Evaluation Report")
    print("  (Simulation only — no quantum advantage claimed)")
    print("═" * 60)
    for k, v in metrics.items():
        print(f"  {k.capitalize():<12}: {v:.4f}")
    print()
    print(
        classification_report(
            y_test, preds, target_names=["Non-binder", "Binder"]
        )
    )
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────
#
# WHY weights-only saving:
# VQC._get_interpret() creates a local `parity` closure that Python's pickle
# cannot serialise (PicklingError).  Instead we extract the trained weight
# vector (a plain numpy array) plus the architecture hyper-parameters, save
# them with numpy, and reconstruct the identical VQC on load — then inject
# the weights so predict() works without re-training.

def save_qml_model(
    vqc,
    path: Path = MODELS_DIR / "qml_model_weights.npz",
) -> None:
    """
    Save VQC trained weights + architecture config as a numpy .npz file.
    Avoids pickling the VQC object (which contains an unpicklable local lambda).
    """
    weights = vqc.weights   # numpy array of optimised circuit parameters
    np.savez(
        str(path),
        weights=weights,
        num_features=np.array(NUM_FEATURES),
        reps=np.array(VQC_REPS),
    )
    logger.info(f"QML weights saved → {path}  ({len(weights)} parameters)")


def load_qml_model(
    path: Path = MODELS_DIR / "qml_model_weights.npz",
):
    """
    Rebuild VQC from saved weights.
    Injects the trained weight vector directly into the model's internal
    OptimizeResult so predict() works without re-running SPSA.
    """
    from scipy.optimize import OptimizeResult

    # Accept both the .npz path and the old .pkl path name for compatibility
    path = Path(path)
    if not path.exists():
        # try swapping extension
        alt = path.with_suffix(".npz") if path.suffix == ".pkl" else path.with_suffix(".pkl")
        if alt.exists():
            path = alt
        else:
            raise FileNotFoundError(
                f"QML model not found at {path}. "
                "Run `python qml_model.py` or `python train.py` first."
            )

    bundle       = np.load(str(path))
    weights      = bundle["weights"]
    num_features = int(bundle["num_features"])
    reps         = int(bundle["reps"])

    # Rebuild the same VQC architecture (max_iter=0 — no re-training)
    vqc = build_vqc(num_features=num_features, reps=reps, max_iter=0)

    # Inject trained weights so the model acts as if it has been fitted
    vqc._fit_result = OptimizeResult(
        x=weights, fun=0.0, nit=0, nfev=0, success=True
    )

    logger.info(f"QML model loaded from {path}  ({len(weights)} parameters)")
    return vqc


# ─────────────────────────────────────────────────────────────────────────────
# Convenience wrapper called by train.py
# ─────────────────────────────────────────────────────────────────────────────

def train_and_save_qml_model(
    X_pca: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.20,
    max_iter: int = SPSA_MAXITER,
) -> Tuple[object, dict]:
    """
    Split, train, evaluate, and save the QML model.
    Returns (vqc, metrics_dict).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=test_size, random_state=42, stratify=y
    )

    vqc     = train_qml_model(X_train, y_train, max_iter=max_iter)
    metrics = evaluate_qml_model(vqc, X_test, y_test)
    save_qml_model(vqc)
    return vqc, metrics


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from preprocessing import load_processed_data

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    data = load_processed_data()
    train_and_save_qml_model(data["X_pca"], data["y"])
