"""
predict.py
==========
Interactive command-line interface for the hybrid drug-discovery pipeline.

Usage
-----
  python predict.py                          # interactive mode
  python predict.py --demo                   # run demo for EGFR + MAP2K7
  python predict.py --protein EGFR           # predict against demo library
  python predict.py --protein MAP2K7 --smiles_file my_compounds.smi

Prediction Modes
----------------
  Classical           : Known protein, RF confidence ≥ threshold, no QML needed
  Hybrid Refinement   : Known protein, high-confidence RF hits → QML refines ranking
  QML Fallback        : Known protein, low RF confidence → QML rescores top candidates
  Exploratory QML     : Unseen protein → QML only, results are NOT validated predictions
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Demo compound library ─────────────────────────────────────────────────────
# A representative set of known kinase inhibitors + inactive controls.
DEMO_SMILES: List[str] = [
    # Erlotinib  — EGFR inhibitor
    "COCCOC1=CC2=C(C=C1OCC)C(=NC=N2)NC3=CC=CC(=C3)C#C",
    # Gefitinib  — EGFR inhibitor
    "COC1=CC2=C(C=C1OCCCN3CCOCC3)C(=NC=N2)NC4=CC(=C(C=C4)F)Cl",
    # Afatinib   — EGFR / HER2 inhibitor
    "CN(C)C/C=C/C(=O)NC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC(=C(C=C3)F)Cl)OC4CCOC4",
    # Vemurafenib — BRAF inhibitor
    "CCCS(=O)(=O)NC1=CC2=C(C=C1)N=C(O2)C3=CC(=NC=C3)C4=C(C=C(C=C4F)Cl)F",
    # Dabrafenib  — BRAF inhibitor
    "CC(C)(C)C1=NC2=C(C=C1)C(=CC=C2)NC(=O)C3=CC(=NC=C3)C4=CC=C(C=C4F)NS(=O)(=O)C",
    # Sorafenib   — BRAF / VEGFR inhibitor
    "CNC(=O)C1=NC=CC(=C1)OC2=CC=C(C=C2)NC(=O)NC3=CC(=C(C=C3)Cl)C(F)(F)F",
    # Trametinib  — MEK1/MEK2 (MAP2K1) inhibitor
    "CC1=C(C(=NO1)C)C(=O)NC2=CC(=C(C=C2F)NC(=O)C3CC3)I",
    # Cobimetinib — MEK inhibitor
    "OC1CC(F)(F)C(NC2=NC3=CC(I)=C(F)C=C3C(=O)N2CC4=CC=CC=C4F)C1",
    # Selumetinib — MEK1/2 inhibitor
    "CC1=CN(C(=O)C(=C1)NC2=CC(=C(C(=C2)Cl)F)I)CC3CCCC3",
    # Axitinib    — VEGFR inhibitor
    "CNC(=O)/C=C/C1=CN=CC(=C1)C2=CC=CS2",
    # Palbociclib — CDK4/6 inhibitor
    "CC1=C2N=C(C(=C2C(=CC1=O)N3CCN(CC3)C)C)C4=CC(=NC=C4)NC5=CC=CC(=C5)F",
    # Ribociclib  — CDK4/6 inhibitor
    "C1CN(CCN1)C2=NC3=C(C=N2)N=CC(=C3)C4=CC(=NC=C4)NC5=CC=CC(=C5)F",
    # Imatinib    — BCR-ABL / PDGFR
    "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5",
    # Sunitinib   — VEGFR / PDGFR
    "CCN(CC)CCNC(=O)C1=C(NC2=CC=C(C=C2F)/C(=C3/C(=O)NC3=O)N)C=C1",
    # Aspirin     — control (non-kinase drug)
    "CC(=O)OC1=CC=CC=C1C(=O)O",
    # Caffeine    — control (non-binder)
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    # Paracetamol — control (non-binder)
    "CC(=O)NC1=CC=C(C=C1)O",
    # Ibuprofen   — control (non-binder)
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    # Metformin   — control (non-binder)
    "CN(C)C(=N)NC(=N)N",
    # Glucose     — control (non-binder)
    "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",
]

BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║   Hybrid Classical + Quantum Drug Discovery Pipeline             ║
║   Cancer Kinase Binding Prediction                               ║
║   Training proteins : EGFR · BRAF · MAPK1 · MAP2K1 · VEGFR2 · CDK2 ║
╚══════════════════════════════════════════════════════════════════╝
"""

MODE_NOTES = {
    "Classical":          "Classical RF — protein in training set, RF scored directly",
    "Hybrid Refinement":  "Classical RF screened; QML refined top candidates",
    "QML Fallback":       "Low RF confidence; QML re-scored top-K candidates",
    "Exploratory QML":    "⚠  UNSEEN protein — QML exploratory only, NOT validated",
}


# ─────────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────────

def display_results(output: dict) -> None:
    """Print a formatted prediction report to stdout."""
    protein   = output["protein"]
    mode      = output["mode"]
    is_known  = output["is_known"]
    df        = output["results"]
    hch       = output["high_confidence_hits"]

    print(BANNER)
    print(f"  Protein Target         : {protein}")
    print(f"  Prediction Mode        : {mode}")
    print(f"  Target in training set : {'YES' if is_known else 'NO'}")
    if is_known:
        print(f"  High-confidence RF hits: {hch}")
    print(f"  Note                   : {MODE_NOTES.get(mode, '')}")
    print()

    # Header
    print(f"{'Rank':<5} {'Binder %':>9} {'Mode label':<24} {'SMILES (truncated)'}")
    print("─" * 80)

    for rank, (_, row) in enumerate(df.iterrows(), start=1):
        score     = row.get("final_score", 0.0)
        if pd.isna(score):
            score = 0.0
        proba_pct = float(score) * 100.0

        # Determine per-compound mode label
        if not is_known:
            m_label = "Exploratory QML"
        elif not pd.isna(row.get("qml_proba", np.nan)):
            m_label = "Hybrid / QML Refined"
        else:
            m_label = "Classical"

        smi   = str(row["smiles"])
        smi_s = (smi[:40] + "…") if len(smi) > 40 else smi
        print(f"{rank:<5} {proba_pct:>8.1f}%  {m_label:<24} {smi_s}")

    print("─" * 80)

    if not is_known:
        print()
        print("  DISCLAIMER: MAP2K7 / this protein was NOT in the training set.")
        print("  Predictions above are EXPLORATORY and have not been validated.")
        print("  Do not use these results as clinical or research conclusions.")

    print()


# ─────────────────────────────────────────────────────────────────────────────
# Interaction modes
# ─────────────────────────────────────────────────────────────────────────────

def run_demo(pipeline) -> None:
    """
    Run automated demonstration for one known protein (EGFR) and
    the unseen target (MAP2K7) to showcase both pipeline branches.
    """
    for protein in ["EGFR", "MAP2K7"]:
        sep = "=" * 68
        print(f"\n{sep}")
        print(f"  DEMO — {protein}")
        print(sep)
        try:
            output = pipeline.predict(protein, DEMO_SMILES)
            display_results(output)
        except Exception as exc:
            print(f"  [Error] {exc}")


def run_single(pipeline, protein: str, smiles_list: List[str]) -> None:
    """Run prediction for a single protein + compound list."""
    try:
        output = pipeline.predict(protein, smiles_list)
        display_results(output)
    except Exception as exc:
        print(f"\n[Error] Prediction failed: {exc}")
        raise


def run_interactive(pipeline) -> None:
    """
    Interactive REPL: user types protein names and optionally provides
    SMILES.  Type 'quit' or 'exit' to stop.
    """
    from preprocessing import TRAINING_PROTEINS, TEST_PROTEIN

    print(BANNER)
    print("  Known training proteins :", ", ".join(sorted(TRAINING_PROTEINS)))
    print("  Unseen test protein     :", ", ".join(TEST_PROTEIN))
    print()

    while True:
        try:
            protein = input("Protein name (or 'quit'): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if protein.lower() in ("quit", "exit", "q", ""):
            print("Exiting.")
            break

        smiles_raw = input(
            "SMILES  (comma-separated, or Enter for demo library): "
        ).strip()

        if smiles_raw:
            smiles_list = [s.strip() for s in smiles_raw.split(",") if s.strip()]
            print(f"Using {len(smiles_list)} provided compounds.")
        else:
            smiles_list = DEMO_SMILES
            print(f"Using demo library ({len(DEMO_SMILES)} compounds).")

        try:
            output = pipeline.predict(protein, smiles_list)
            display_results(output)
        except Exception as exc:
            print(f"\n[Error] {exc}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Hybrid Classical + Quantum Drug Discovery Prediction"
    )
    p.add_argument(
        "--demo",
        action="store_true",
        help="Run demo predictions for EGFR (known) and MAP2K7 (unseen)",
    )
    p.add_argument(
        "--protein",
        type=str,
        default=None,
        help="Target protein name for a single prediction run",
    )
    p.add_argument(
        "--smiles_file",
        type=str,
        default=None,
        help="Path to a file containing SMILES strings (one per line)",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Enable INFO-level logging",
    )
    return p.parse_args()


def main():
    args = _parse_args()

    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    print("Loading models … ", end="", flush=True)
    try:
        from hybrid_pipeline import HybridPipeline
        pipeline = HybridPipeline()
        print("done.\n")
    except FileNotFoundError as exc:
        print(f"\n[Error] {exc}")
        print("\nPlease run the training pipeline first:")
        print("  python train.py")
        sys.exit(1)

    if args.demo:
        run_demo(pipeline)

    elif args.protein:
        if args.smiles_file:
            fpath = Path(args.smiles_file)
            if not fpath.exists():
                print(f"[Error] SMILES file not found: {fpath}")
                sys.exit(1)
            smiles_list = [
                line.strip()
                for line in fpath.read_text().splitlines()
                if line.strip() and not line.startswith("#")
            ]
        else:
            smiles_list = DEMO_SMILES
            print(f"No SMILES file provided — using demo library ({len(DEMO_SMILES)} compounds).")

        run_single(pipeline, args.protein, smiles_list)

    else:
        run_interactive(pipeline)


if __name__ == "__main__":
    main()
