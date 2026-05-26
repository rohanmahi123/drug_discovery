"""
preprocessing.py
================
Full preprocessing pipeline for the hybrid drug-discovery framework.

Steps
-----
1.  Fetch bioactivity data from ChEMBL (with on-disk cache).
2.  Clean missing values and normalise protein names.
3.  Assign binary binder labels  (activity < 1000 nM → 1, else → 0).
4.  Validate SMILES strings with RDKit and drop invalid ones.
5.  Generate Morgan fingerprints  (radius=2, nBits=2048).
6.  Fit StandardScaler + PCA (2048 → 8 dimensions) on training data.
7.  Persist preprocessors and processed arrays for downstream models.
"""

import os
import pickle
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# RDKit cheminformatics
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

# Scikit-learn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ChEMBL REST client
from chembl_webresource_client.new_client import new_client

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")   # suppress noisy RDKit messages

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Cancer-related kinase proteins used for TRAINING (MAP2K7 is deliberately excluded)
TRAINING_PROTEINS: Dict[str, str] = {
    "EGFR":   "CHEMBL203",   # Epidermal growth factor receptor
    "BRAF":   "CHEMBL5145",  # Serine/threonine-protein kinase B-raf
    "MAPK1":  "CHEMBL3234",  # Mitogen-activated protein kinase 1 (ERK2)
    "MAP2K1": "CHEMBL2409",  # MEK1
    "VEGFR2": "CHEMBL279",   # Vascular endothelial growth factor receptor 2
    "CDK2":   "CHEMBL301",   # Cyclin-dependent kinase 2
}

# Held-out unseen target — used only for testing, NEVER for training
TEST_PROTEIN: Dict[str, str] = {
    "MAP2K7": "CHEMBL3885",  # MEK7 — unseen during training
}

BINDING_THRESHOLD_NM: float = 1000.0   # compounds below this (nM) → binder
ACTIVITY_TYPES: Tuple[str, ...] = ("IC50", "Ki", "Kd")

MORGAN_RADIUS: int = 2
MORGAN_NBITS: int = 2048
PCA_COMPONENTS: int = 8

# Cap records per target per activity type to keep dataset manageable
MAX_RECORDS_PER_TYPE: int = 400

DATA_DIR   = Path("data")
MODELS_DIR = Path("models")
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Data Fetching
# ─────────────────────────────────────────────────────────────────────────────

def _find_chembl_id_by_name(protein_name: str) -> Optional[str]:
    """
    Search ChEMBL target database by protein name.
    Falls back to None if no human single-protein match is found.
    """
    try:
        target_client = new_client.target
        results = list(
            target_client.filter(
                pref_name__icontains=protein_name,
                target_type="SINGLE PROTEIN",
                organism="Homo sapiens",
            ).only("target_chembl_id", "pref_name")[:3]
        )
        if results:
            cid = results[0]["target_chembl_id"]
            logger.info(f"  Found {protein_name} → {cid} ({results[0]['pref_name']})")
            return cid
    except Exception as exc:
        logger.warning(f"  ChEMBL search failed for {protein_name}: {exc}")
    return None


def fetch_chembl_activities(
    chembl_id: str,
    target_name: str,
    max_per_type: int = MAX_RECORDS_PER_TYPE,
) -> pd.DataFrame:
    """
    Query ChEMBL bioactivity endpoint for one target.
    Filters for binding assays with numeric nM values.
    Returns a raw DataFrame ready for cleaning.
    """
    activity_client = new_client.activity
    records: List[dict] = []

    for atype in ACTIVITY_TYPES:
        try:
            hits = list(
                activity_client.filter(
                    target_chembl_id=chembl_id,
                    standard_type=atype,
                    standard_relation__in=["=", "<", "<="],
                    standard_units="nM",
                    assay_type="B",          # binding assay only
                ).only(
                    "molecule_chembl_id",
                    "canonical_smiles",
                    "standard_type",
                    "standard_value",
                    "standard_units",
                )[:max_per_type]
            )
            for rec in hits:
                records.append(
                    {
                        "target_name":    target_name,
                        "chembl_id":      chembl_id,
                        "molecule_id":    rec.get("molecule_chembl_id", ""),
                        "smiles":         rec.get("canonical_smiles", ""),
                        "activity_type":  rec.get("standard_type", ""),
                        "activity_value": rec.get("standard_value", None),
                    }
                )
            logger.info(f"    {target_name}/{atype}: {len(hits)} records")
        except Exception as exc:
            logger.warning(f"    Could not fetch {atype} for {target_name}: {exc}")

    return pd.DataFrame(records)


def fetch_all_training_data(
    cache_path: Optional[Path] = None,
    force_refetch: bool = False,
) -> pd.DataFrame:
    """
    Fetch bioactivity data for all TRAINING proteins.
    Results are cached locally to avoid repeated API calls.
    """
    cache_path = cache_path or DATA_DIR / "raw_training_data.csv"

    if cache_path.exists() and not force_refetch:
        logger.info(f"Loading cached training data from {cache_path}")
        return pd.read_csv(cache_path)

    frames: List[pd.DataFrame] = []
    for name, cid in TRAINING_PROTEINS.items():
        logger.info(f"Fetching {name} ({cid}) …")
        df = fetch_chembl_activities(cid, name)

        # If hardcoded ID returned nothing, try name-based search
        if len(df) == 0:
            logger.info(f"  Trying name-based search for {name} …")
            found_cid = _find_chembl_id_by_name(name)
            if found_cid and found_cid != cid:
                df = fetch_chembl_activities(found_cid, name)

        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(cache_path, index=False)
    logger.info(f"Saved {len(combined)} raw records → {cache_path}")
    return combined


def fetch_test_target_data(
    cache_path: Optional[Path] = None,
    force_refetch: bool = False,
) -> pd.DataFrame:
    """
    Fetch MAP2K7 data for held-out evaluation.
    This data must NEVER be used during training.
    """
    cache_path = cache_path or DATA_DIR / "raw_test_data.csv"

    if cache_path.exists() and not force_refetch:
        logger.info(f"Loading cached test data from {cache_path}")
        return pd.read_csv(cache_path)

    target_name, target_id = next(iter(TEST_PROTEIN.items()))
    logger.info(f"Fetching held-out test target {target_name} ({target_id}) …")
    df = fetch_chembl_activities(target_id, target_name)

    if len(df) == 0:
        found_cid = _find_chembl_id_by_name(target_name)
        if found_cid:
            df = fetch_chembl_activities(found_cid, target_name)

    df.to_csv(cache_path, index=False)
    logger.info(f"Saved {len(df)} MAP2K7 records → {cache_path}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Cleaning & Label Generation
# ─────────────────────────────────────────────────────────────────────────────

def clean_bioactivity_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with missing SMILES or activity values,
    coerce types, and normalise protein names to uppercase.
    """
    df = df.copy()

    # Normalise protein names
    df["target_name"] = df["target_name"].str.upper().str.strip()

    # Coerce numeric activity
    df["activity_value"] = pd.to_numeric(df["activity_value"], errors="coerce")

    # Strip SMILES whitespace
    df["smiles"] = df["smiles"].astype(str).str.strip()

    before = len(df)
    df = df.dropna(subset=["smiles", "activity_value"])
    df = df[df["smiles"].str.len() > 3]          # remove degenerate strings
    df = df[df["activity_value"] > 0]            # remove non-positive values
    df = df.drop_duplicates(subset=["smiles", "target_name"])

    logger.info(f"Cleaning: {before} → {len(df)} records after dedup/NaN removal")
    return df.reset_index(drop=True)


def create_binary_labels(
    df: pd.DataFrame,
    threshold_nm: float = BINDING_THRESHOLD_NM,
) -> pd.DataFrame:
    """
    Binder  (label=1): activity_value < threshold_nm
    Non-binder (label=0): activity_value >= threshold_nm

    1000 nM is a widely-used hit threshold in kinase drug discovery.
    """
    df = df.copy()
    df["label"] = (df["activity_value"] < threshold_nm).astype(int)
    binder_pct = df["label"].mean() * 100
    logger.info(
        f"Labels: {binder_pct:.1f}% binders | {100 - binder_pct:.1f}% non-binders"
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3.  SMILES Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_and_filter_smiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate every SMILES string using RDKit.
    Rows whose SMILES cannot be parsed are silently dropped.
    """
    valid: List[bool] = []
    for smi in tqdm(df["smiles"], desc="Validating SMILES", leave=False):
        mol = Chem.MolFromSmiles(str(smi))
        valid.append(mol is not None)

    before = len(df)
    df = df[valid].reset_index(drop=True)
    logger.info(
        f"SMILES validation: {before} → {len(df)} valid molecules "
        f"({before - len(df)} invalid dropped)"
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Morgan Fingerprint Generation
# ─────────────────────────────────────────────────────────────────────────────

def smiles_to_morgan_fingerprints(
    smiles_list: List[str],
    radius: int = MORGAN_RADIUS,
    n_bits: int = MORGAN_NBITS,
) -> np.ndarray:
    """
    Convert a list of SMILES to a 2-D binary fingerprint matrix.
    Shape: (n_compounds, n_bits).
    Compounds whose SMILES fail are represented as all-zero vectors.
    """
    fps: List[np.ndarray] = []
    for smi in tqdm(smiles_list, desc="Morgan fingerprints", leave=False):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, radius=radius, nBits=n_bits
            )
            fps.append(np.array(fp, dtype=np.float32))
        else:
            fps.append(np.zeros(n_bits, dtype=np.float32))
    return np.array(fps)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Feature Scaling + PCA
# ─────────────────────────────────────────────────────────────────────────────

def fit_scaling_pca(
    X: np.ndarray,
    n_components: int = PCA_COMPONENTS,
) -> Tuple[StandardScaler, PCA, np.ndarray]:
    """
    Fit StandardScaler then PCA on training fingerprints.
    Returns (fitted scaler, fitted PCA, transformed X).

    Binary Morgan fingerprints benefit from centering before PCA so that
    rarely set bits don't dominate the principal axes.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    explained_var = pca.explained_variance_ratio_.sum() * 100
    logger.info(
        f"PCA: {X.shape[1]}D → {n_components}D | "
        f"explained variance: {explained_var:.1f}%"
    )
    return scaler, pca, X_pca


def apply_scaling_pca(
    X: np.ndarray,
    scaler: StandardScaler,
    pca: PCA,
) -> np.ndarray:
    """Apply pre-fitted scaler + PCA to new (test/predict) data."""
    return pca.transform(scaler.transform(X))


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Persistence Helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_preprocessors(
    scaler: StandardScaler,
    pca: PCA,
    path: Path = MODELS_DIR / "preprocessors.pkl",
) -> None:
    with open(path, "wb") as fh:
        pickle.dump({"scaler": scaler, "pca": pca}, fh)
    logger.info(f"Preprocessors saved → {path}")


def load_preprocessors(
    path: Path = MODELS_DIR / "preprocessors.pkl",
) -> Tuple[StandardScaler, PCA]:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Preprocessors not found at {path}. Run the training pipeline first."
        )
    with open(path, "rb") as fh:
        bundle = pickle.load(fh)
    return bundle["scaler"], bundle["pca"]


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Pipeline Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run_preprocessing_pipeline(force_refetch: bool = False) -> Dict:
    """
    Run the complete preprocessing pipeline end-to-end.

    Returns a dict containing:
        X_fp          : Morgan fingerprints (n_compounds, 2048)
        X_pca         : PCA-reduced features (n_compounds, 8)
        y             : binary labels (n_compounds,)
        smiles        : array of SMILES strings
        target_names  : array of protein names
        scaler        : fitted StandardScaler
        pca           : fitted PCA
        df            : fully processed DataFrame
    """
    cache_proc_npz = DATA_DIR / "processed_training_data.npz"
    cache_proc_csv = DATA_DIR / "processed_training_data.csv"

    # ── Step 1: Fetch ─────────────────────────────────────────────────────────
    df_raw = fetch_all_training_data(force_refetch=force_refetch)

    # ── Step 2: Clean + label ─────────────────────────────────────────────────
    df = clean_bioactivity_data(df_raw)
    df = create_binary_labels(df)

    # ── Step 3: Validate SMILES ───────────────────────────────────────────────
    df = validate_and_filter_smiles(df)

    # ── Step 4: Morgan fingerprints ───────────────────────────────────────────
    logger.info("Generating Morgan fingerprints …")
    X_fp = smiles_to_morgan_fingerprints(df["smiles"].tolist())
    y    = df["label"].values.astype(int)
    target_names = df["target_name"].values

    # ── Step 5: Scale + PCA ───────────────────────────────────────────────────
    logger.info("Fitting StandardScaler + PCA …")
    scaler, pca, X_pca = fit_scaling_pca(X_fp)
    save_preprocessors(scaler, pca)

    # ── Step 6: Persist ───────────────────────────────────────────────────────
    np.savez_compressed(
        cache_proc_npz,
        X_fp=X_fp,
        X_pca=X_pca,
        y=y,
        target_names=target_names,
    )
    df.to_csv(cache_proc_csv, index=False)
    logger.info(
        f"Preprocessing complete.  "
        f"Shape: {X_pca.shape}  |  Binders: {y.mean():.1%}"
    )

    return {
        "X_fp":         X_fp,
        "X_pca":        X_pca,
        "y":            y,
        "smiles":       df["smiles"].values,
        "target_names": target_names,
        "scaler":       scaler,
        "pca":          pca,
        "df":           df,
    }


def load_processed_data() -> Dict:
    """
    Load previously processed data from disk.
    Raises FileNotFoundError if the pipeline has not been run yet.
    """
    npz_path = DATA_DIR / "processed_training_data.npz"
    csv_path = DATA_DIR / "processed_training_data.csv"

    if not npz_path.exists():
        raise FileNotFoundError(
            "Processed data not found.  "
            "Run `python preprocessing.py` or `python train.py` first."
        )

    bundle = np.load(npz_path, allow_pickle=True)
    df     = pd.read_csv(csv_path)
    scaler, pca = load_preprocessors()

    return {
        "X_fp":         bundle["X_fp"],
        "X_pca":        bundle["X_pca"],
        "y":            bundle["y"].astype(int),
        "smiles":       df["smiles"].values,
        "target_names": bundle["target_names"],
        "scaler":       scaler,
        "pca":          pca,
        "df":           df,
    }


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    result = run_preprocessing_pipeline()
    print(
        f"\nDataset ready: {result['X_pca'].shape[0]} compounds, "
        f"{result['X_pca'].shape[1]} PCA features"
    )
    print(f"Binder ratio: {result['y'].mean():.2%}")
    print("Per-target distribution:")
    df_out = result["df"]
    print(df_out.groupby("target_name")["label"].value_counts().unstack(fill_value=0))
