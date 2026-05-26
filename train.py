"""
train.py
========
Orchestrates the complete training pipeline:

  Step 1 — Preprocessing   : fetch ChEMBL data, generate fingerprints, PCA
  Step 2 — Classical model  : train + evaluate RandomForest
  Step 3 — QML model        : train + evaluate VQC  (can be skipped with --skip-qml)
  Step 4 — Evaluation       : hybrid pipeline evaluation on MAP2K7 (unseen target)

Flags
-----
  --skip-qml       Skip the (slow) QML training step.
                   Useful for quick iteration on Classical model only.
  --force-refetch  Re-download ChEMBL data even if local cache exists.
  --qml-iters N    Override SPSA max iterations (default: 100).
                   Use 30–50 for a fast sanity check; 200+ for better convergence.
  --eval-map2k7    Run MAP2K7 exploratory evaluation after training.
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DIVIDER = "═" * 60


def run_training_pipeline(
    skip_qml:      bool = False,
    force_refetch: bool = False,
    qml_iters:     int  = 100,
    eval_map2k7:   bool = True,
) -> None:

    # ══════════════════════════════════════════════════════════════════════
    # STEP 1 — Preprocessing
    # ══════════════════════════════════════════════════════════════════════
    logger.info(DIVIDER)
    logger.info("STEP 1 / Preprocessing")
    logger.info(DIVIDER)

    from preprocessing import run_preprocessing_pipeline
    data = run_preprocessing_pipeline(force_refetch=force_refetch)

    logger.info(
        f"Dataset ready: {data['X_pca'].shape[0]} compounds | "
        f"binder rate: {data['y'].mean():.1%}"
    )

    # ══════════════════════════════════════════════════════════════════════
    # STEP 2 — Classical Model
    # ══════════════════════════════════════════════════════════════════════
    logger.info(DIVIDER)
    logger.info("STEP 2 / Classical Model (RandomForest)")
    logger.info(DIVIDER)

    from classical_model import train_and_save_classical_model
    model, threshold, classical_metrics = train_and_save_classical_model(
        data["X_fp"],   # full 2048-dim Morgan fingerprints
        data["y"],
    )
    logger.info(
        f"Classical model saved.  "
        f"Threshold={threshold:.2f}  |  Recall={classical_metrics['recall']:.4f}"
    )

    # ══════════════════════════════════════════════════════════════════════
    # STEP 3 — QML Model (optional)
    # ══════════════════════════════════════════════════════════════════════
    if not skip_qml:
        logger.info(DIVIDER)
        logger.info("STEP 3 / Quantum ML Model (VQC)")
        logger.info(f"  SPSA max iterations : {qml_iters}")
        logger.info(f"  Training budget     : up to 500 balanced samples")
        logger.info("  This step may take 10–60 minutes on CPU …")
        logger.info(DIVIDER)

        from qml_model import train_and_save_qml_model
        vqc, qml_metrics = train_and_save_qml_model(
            data["X_pca"],   # 8-dim PCA features
            data["y"],
            max_iter=qml_iters,
        )
        logger.info(
            f"QML model saved.  "
            f"Accuracy={qml_metrics['accuracy']:.4f}  |  F1={qml_metrics['f1']:.4f}"
        )
    else:
        logger.info("STEP 3 / QML training SKIPPED  (--skip-qml)")
        # Check that a saved QML model already exists for prediction
        qml_path = Path("models") / "qml_model.pkl"
        if not qml_path.exists():
            logger.warning(
                f"No QML model found at {qml_path}.  "
                "predict.py will fail until QML model is trained."
            )

    # ══════════════════════════════════════════════════════════════════════
    # STEP 4 — Hybrid evaluation on MAP2K7 (unseen target)
    # ══════════════════════════════════════════════════════════════════════
    if eval_map2k7 and not skip_qml:
        logger.info(DIVIDER)
        logger.info("STEP 4 / Hybrid Evaluation on MAP2K7  (held-out unseen target)")
        logger.info(DIVIDER)

        try:
            from hybrid_pipeline import HybridPipeline
            pipeline = HybridPipeline()
            pipeline.evaluate_on_test_target()
        except Exception as exc:
            logger.warning(f"MAP2K7 evaluation failed: {exc}")
    else:
        logger.info("STEP 4 / MAP2K7 evaluation skipped.")

    logger.info(DIVIDER)
    logger.info("Training pipeline complete.")
    logger.info("Run  `python predict.py --demo`  to test predictions.")
    logger.info(DIVIDER)


# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Hybrid Classical + Quantum Drug Discovery — Training Pipeline"
    )
    p.add_argument(
        "--skip-qml",
        action="store_true",
        help="Skip QML training (fast mode for Classical model only)",
    )
    p.add_argument(
        "--force-refetch",
        action="store_true",
        help="Re-download ChEMBL data even if local cache exists",
    )
    p.add_argument(
        "--qml-iters",
        type=int,
        default=100,
        metavar="N",
        help="SPSA max iterations for VQC training (default: 100)",
    )
    p.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip MAP2K7 evaluation after training",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_training_pipeline(
        skip_qml      = args.skip_qml,
        force_refetch = args.force_refetch,
        qml_iters     = args.qml_iters,
        eval_map2k7   = not args.no_eval,
    )
