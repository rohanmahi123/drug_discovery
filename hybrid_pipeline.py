"""
hybrid_pipeline.py
==================
Decision logic that combines Classical RF and Quantum VQC predictions.

Flow
----

  ┌─────────────────────────────────────────────────────────────────────┐
  │  User provides: protein_name + list[SMILES]                         │
  └───────────────────────────┬─────────────────────────────────────────┘
                              │
               ┌──────────────▼──────────────┐
               │  Is protein in training set? │
               └──────────────┬──────────────┘
              YES              │              NO
               │               │               │
               ▼               │               ▼
  Classical RF screening       │    ┌─────────────────────┐
  (all compounds ranked)       │    │  Exploratory QML    │
               │               │    │  (no classical step)│
     ┌─────────▼──────────┐    │    └──────────┬──────────┘
     │ confidence ≥ 0.70? │    │               │
     └─────────┬──────────┘    │               │
          YES  │  NO           │               │
          │    │               │               │
          ▼    ▼               │               │
    QML Hybrid  QML Fallback   │               │
    Refinement  (top-K)        │               │
          │    │               │               │
          └────┴───────────────┘───────────────┘
                              │
                              ▼
                  Ranked compound list
                  + prediction mode label
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from preprocessing import (
    TRAINING_PROTEINS,
    TEST_PROTEIN,
    apply_scaling_pca,
    load_preprocessors,
    smiles_to_morgan_fingerprints,
    validate_and_filter_smiles,
)
from classical_model import load_classical_model, predict_with_proba
from qml_model import load_qml_model, predict_qml, rescale_to_pi, MODELS_DIR as QML_MODELS_DIR

logger = logging.getLogger(__name__)

# ── Tunable hyper-parameters ──────────────────────────────────────────────────
CONFIDENCE_THRESHOLD: float = 0.70  # RF proba ≥ this → "high-confidence" hit
MIN_HIGH_CONF_HITS:   int   = 5     # need ≥ this many hits to choose Hybrid mode
TOP_K_CLASSICAL:      int   = 20    # top-K from RF passed to QML for refinement
TOP_K_DISPLAY:        int   = 10    # rows shown in the final result table

# Weighted ensemble: quantum model contributes more weight during refinement
RF_WEIGHT:  float = 0.40
QML_WEIGHT: float = 0.60


class HybridPipeline:
    """
    Orchestrates Classical and Quantum models based on target familiarity.

    Usage
    -----
    pipeline = HybridPipeline()
    result   = pipeline.predict("EGFR",   smiles_list)  # known protein
    result   = pipeline.predict("MAP2K7", smiles_list)  # unseen protein
    """

    def __init__(self) -> None:
        self.scaler, self.pca              = load_preprocessors()
        self.classical_model, self.threshold = load_classical_model()
        self.qml_model = load_qml_model(QML_MODELS_DIR / "qml_model_weights.npz")

        # Set of protein names seen during training (upper-cased for comparison)
        self._known_proteins: set = {k.upper() for k in TRAINING_PROTEINS}

        logger.info("HybridPipeline initialised with all models loaded.")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _is_known_protein(self, protein_name: str) -> bool:
        return protein_name.upper() in self._known_proteins

    def _featurise(
        self, smiles_list: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Validate SMILES, generate Morgan fingerprints, apply scaling+PCA.
        Returns (X_fp, X_pca, valid_smiles).
        Invalid SMILES are removed and the caller receives the clean list.
        """
        # Filter SMILES using a temporary DataFrame
        tmp_df = pd.DataFrame({"smiles": smiles_list})
        tmp_df = validate_and_filter_smiles(tmp_df)
        valid_smiles = tmp_df["smiles"].tolist()

        if not valid_smiles:
            raise ValueError("No valid SMILES provided after RDKit validation.")

        X_fp  = smiles_to_morgan_fingerprints(valid_smiles)
        X_pca = apply_scaling_pca(X_fp, self.scaler, self.pca)
        return X_fp, X_pca, valid_smiles

    # ── Classical screening ───────────────────────────────────────────────────

    def _classical_screen(
        self,
        valid_smiles: List[str],
        X_fp: np.ndarray,
    ) -> pd.DataFrame:
        """
        Score every compound with the RandomForest.
        Returns a DataFrame sorted by rf_proba descending.
        """
        preds, probas = predict_with_proba(
            self.classical_model, X_fp, self.threshold
        )
        df = pd.DataFrame(
            {
                "smiles":   valid_smiles,
                "rf_pred":  preds,
                "rf_proba": probas,
            }
        )
        return df.sort_values("rf_proba", ascending=False).reset_index(drop=True)

    # ── QML refinement / exploration ─────────────────────────────────────────

    def _qml_score(
        self,
        smiles_subset: List[str],
        X_pca_subset: np.ndarray,
    ) -> pd.DataFrame:
        """
        Run QML on a (small) candidate set.
        Returns DataFrame with qml_pred and qml_proba columns.
        """
        X_norm = rescale_to_pi(X_pca_subset)
        qml_preds, qml_probas = predict_qml(
            self.qml_model, X_norm, apply_rescale=False
        )
        return pd.DataFrame(
            {
                "smiles":    smiles_subset,
                "qml_pred":  qml_preds,
                "qml_proba": qml_probas,
            }
        )

    # ── Decision logic ────────────────────────────────────────────────────────

    def predict(
        self,
        protein_name: str,
        smiles_list: List[str],
    ) -> Dict:
        """
        Main prediction entry point.

        Parameters
        ----------
        protein_name : str
            Name of the target kinase protein.
        smiles_list  : list[str]
            SMILES strings of compounds to screen.

        Returns
        -------
        dict with keys:
            protein              : normalised protein name
            mode                 : prediction mode label
            is_known             : True if protein was in training set
            results              : pd.DataFrame ranked by final_score
            high_confidence_hits : int (0 for unseen targets)
        """
        protein_upper  = protein_name.strip().upper()
        is_known       = self._is_known_protein(protein_upper)
        high_conf_hits = 0

        # ── Featurise all compounds ────────────────────────────────────────
        X_fp, X_pca, valid_smiles = self._featurise(smiles_list)
        n_compounds = len(valid_smiles)
        logger.info(
            f"Predict: protein={protein_upper}  |  "
            f"{n_compounds} valid compounds  |  known={is_known}"
        )

        # ══════════════════════════════════════════════════════════════════
        # KNOWN PROTEIN  →  Classical screening + optional QML refinement
        # ══════════════════════════════════════════════════════════════════
        if is_known:
            classical_df = self._classical_screen(valid_smiles, X_fp)

            high_conf_hits = int(
                (classical_df["rf_proba"] >= CONFIDENCE_THRESHOLD).sum()
            )

            # Decide mode based on how many high-confidence hits RF found
            if high_conf_hits >= MIN_HIGH_CONF_HITS:
                mode = "Hybrid Refinement"
            else:
                mode = "QML Fallback"

            # Select top-K candidates from RF ranking for QML
            top_k   = min(TOP_K_CLASSICAL, n_compounds)
            top_df  = classical_df.head(top_k).copy()
            top_idx = top_df.index.tolist()

            logger.info(
                f"Mode: {mode}  |  "
                f"high-conf={high_conf_hits}  |  "
                f"sending top-{top_k} to QML …"
            )

            top_smiles = top_df["smiles"].tolist()
            top_X_pca  = X_pca[top_idx]

            qml_df = self._qml_score(top_smiles, top_X_pca)

            # Merge and compute ensemble score
            results = top_df.merge(
                qml_df[["smiles", "qml_pred", "qml_proba"]],
                on="smiles",
                how="left",
            )
            # Weighted ensemble of RF and QML probabilities
            results["final_score"] = (
                RF_WEIGHT  * results["rf_proba"]
                + QML_WEIGHT * results["qml_proba"].fillna(results["rf_proba"])
            )

        # ══════════════════════════════════════════════════════════════════
        # UNSEEN PROTEIN  →  Exploratory QML only
        # ══════════════════════════════════════════════════════════════════
        else:
            mode = "Exploratory QML"
            logger.info(
                f"{protein_upper} is NOT in training set → "
                "Exploratory QML mode (no Classical screening)"
            )

            qml_df = self._qml_score(valid_smiles, X_pca)
            results = qml_df.copy()
            results["rf_pred"]    = np.nan
            results["rf_proba"]   = np.nan
            results["final_score"] = results["qml_proba"]

        # ── Sort and trim ────────────────────────────────────────────────
        results = results.sort_values(
            "final_score", ascending=False
        ).reset_index(drop=True)

        return {
            "protein":              protein_upper,
            "mode":                 mode,
            "is_known":             is_known,
            "results":              results.head(TOP_K_DISPLAY),
            "high_confidence_hits": high_conf_hits,
        }

    # ── Held-out evaluation on MAP2K7 ─────────────────────────────────────────

    def evaluate_on_test_target(
        self, max_compounds: int = 500
    ) -> Dict[str, float]:
        """
        Evaluate the Exploratory QML branch on the held-out MAP2K7 target.

        IMPORTANT:  MAP2K7 was never included in training.  The metrics here
        reflect how well QML generalises to an unseen kinase, not in-distribution
        accuracy.  Do not interpret these numbers as validated prediction accuracy.
        """
        from preprocessing import (
            clean_bioactivity_data,
            create_binary_labels,
            fetch_test_target_data,
            validate_and_filter_smiles,
        )
        from sklearn.metrics import (
            accuracy_score,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
        )

        logger.info("Fetching MAP2K7 held-out data …")
        df = fetch_test_target_data()
        df = clean_bioactivity_data(df)
        df = create_binary_labels(df)
        df = validate_and_filter_smiles(df)

        if len(df) == 0:
            logger.warning("No valid MAP2K7 compounds — skipping evaluation.")
            return {}

        df = df.head(max_compounds)
        smiles = df["smiles"].tolist()
        y_true = df["label"].values

        output = self.predict("MAP2K7", smiles)
        result_df = output["results"]

        # Align predictions with ground-truth labels using SMILES as key
        score_map = dict(zip(result_df["smiles"], result_df["final_score"]))
        y_score   = np.array([score_map.get(s, 0.0) for s in smiles])
        y_pred    = (y_score >= 0.50).astype(int)

        metrics = {
            "accuracy":  accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall":    recall_score(y_true, y_pred, zero_division=0),
            "f1":        f1_score(y_true, y_pred, zero_division=0),
        }

        print("\n" + "═" * 60)
        print("  MAP2K7 Exploratory Evaluation")
        print("  ⚠  Unseen target — exploratory QML, NOT validated accuracy")
        print("═" * 60)
        for k, v in metrics.items():
            print(f"  {k.capitalize():<12}: {v:.4f}")
        print()
        print("  Confusion Matrix (rows=true, cols=predicted):")
        cm = confusion_matrix(y_true, y_pred)
        print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
        print(f"  FN={cm[1,0]}  TP={cm[1,1]}")
        return metrics


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )
    pipeline = HybridPipeline()
    pipeline.evaluate_on_test_target()
