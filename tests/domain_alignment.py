"""
domain_alignment.py  --  Module 3 (Bamelaai Marbaniang)
CORAL (Correlation Alignment) and the before/after evaluation driver.

CORAL (Section 5.1):
    C_S = cov(X_S) + lambda*I ,   C_T = cov(X_T) + lambda*I
    X_S_aligned = X_S . C_S^(-1/2) . C_T^(1/2)
Matrix powers are computed by eigendecomposition  A = U S U^T -> A^p = U S^p U^T.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.divergence import divergence_report

LABEL_COL = "label"
_MIN_EIG = 1e-12


# --------------------------------------------------------------------------- #
# CORAL
# --------------------------------------------------------------------------- #
def _sym_matrix_power(C: np.ndarray, power: float) -> np.ndarray:
    """C^power for a symmetric PSD matrix via eigendecomposition."""
    eigvals, U = np.linalg.eigh(C)
    eigvals = np.clip(eigvals, _MIN_EIG, None)
    return (U * eigvals ** power) @ U.T


def coral_align(
    X_source: np.ndarray,
    X_target: np.ndarray,
    lambda_reg: float = 1.0,
    match_mean: bool = False,
) -> np.ndarray:
    """
    Align the source covariance to the target covariance (CORAL).

    Parameters
    ----------
    X_source, X_target : (n_s, d) and (n_t, d) arrays (same d).
    lambda_reg : ridge term added to both covariances (spec default 1.0).
        Larger -> more stable but weaker alignment; try 1e-3 for near-exact
        covariance matching.
    match_mean : if True, also shift the aligned source so its mean equals the
        target mean (CORAL itself only matches second-order statistics).

    Returns
    -------
    X_source_aligned, same shape as X_source.
    """
    Xs = np.asarray(X_source, dtype=np.float64)
    Xt = np.asarray(X_target, dtype=np.float64)
    if Xs.ndim != 2 or Xt.ndim != 2 or Xs.shape[1] != Xt.shape[1]:
        raise ValueError("X_source and X_target must be 2-D with the same number of columns.")

    d = Xs.shape[1]
    eye = np.eye(d)
    C_s = np.cov(Xs, rowvar=False) + lambda_reg * eye       # step 1
    C_t = np.cov(Xt, rowvar=False) + lambda_reg * eye       # step 2
    C_s_inv_sqrt = _sym_matrix_power(C_s, -0.5)             # step 3
    C_t_sqrt = _sym_matrix_power(C_t, 0.5)                  # step 4

    X_aligned = Xs @ C_s_inv_sqrt @ C_t_sqrt                # step 5 (whiten, re-colour)

    if match_mean:
        X_aligned = X_aligned - X_aligned.mean(axis=0) + Xt.mean(axis=0)
    return X_aligned


# --------------------------------------------------------------------------- #
# Cross-dataset evaluation (Member 2's function, with a stand-in fallback)
# --------------------------------------------------------------------------- #
def _fallback_cross_eval(df_source, df_target, source_name, target_name) -> pd.DataFrame:
    """
    Minimal stand-in used ONLY if src.cross_dataset is not merged yet.
    Same protocol: 80% stratified source train, scaler fit on source train,
    test on 100% of target. Same output columns as Member 2's function.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                                 precision_score, recall_score)
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    feats = [c for c in df_source.columns if c != LABEL_COL]
    Xtr, _, ytr, _ = train_test_split(df_source[feats], df_source[LABEL_COL],
                                      test_size=0.2, stratify=df_source[LABEL_COL],
                                      random_state=42)
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(df_target[feats])
    yte = df_target[LABEL_COL]
    models = {
        "RF": RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1),
        "DT": DecisionTreeClassifier(max_depth=20, random_state=42),
        "LR": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5, metric="minkowski"),
        "SVM": SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42),
    }
    rows = []
    for name, m in models.items():
        pred = m.fit(Xtr_s, ytr).predict(Xte_s)
        rows.append({
            "model": name, "train_dataset": source_name, "test_dataset": target_name,
            "accuracy": accuracy_score(yte, pred),
            "precision": precision_score(yte, pred, zero_division=0),
            "recall": recall_score(yte, pred, zero_division=0),
            "f1": f1_score(yte, pred, zero_division=0),
            "mcc": matthews_corrcoef(yte, pred),
        })
    return pd.DataFrame(rows)


def _cross_eval(df_source, df_target, source_name, target_name) -> pd.DataFrame:
    try:
        from src.cross_dataset import run_cross_dataset_evaluation
    except ImportError:
        print("[domain_alignment] src.cross_dataset not found -> using stand-in evaluator.")
        return _fallback_cross_eval(df_source, df_target, source_name, target_name)
    return run_cross_dataset_evaluation(df_source, df_target, source_name, target_name)


# --------------------------------------------------------------------------- #
# Align + evaluate
# --------------------------------------------------------------------------- #
def align_and_evaluate(
    df_source: pd.DataFrame,
    df_target: pd.DataFrame,
    source_name: str,
    target_name: str,
    lambda_reg: float = 1.0,
) -> dict:
    """
    1. Standardise both domains with a scaler fitted on the SOURCE only
       (same rule as the cross-dataset protocol), then CORAL-align the source
       features to the target (with mean matching).
    2. Re-run the cross-dataset evaluation on the aligned data.
    3. Compute divergence metrics before and after alignment (both measured in
       the same standardised space so they are comparable).

    Both inputs are mapped DataFrames (unified features + "label").

    Returns
    -------
    {"results_before": DataFrame, "results_after": DataFrame,
     "divergence_before": dict,   "divergence_after": dict,
     "covariance_shift": {"before": float, "after": float,
                          "cov_source": ndarray, "cov_aligned": ndarray,
                          "cov_target": ndarray, "features": list[str]},
     "aligned_source": DataFrame}
    """
    feats = [c for c in df_source.columns if c != LABEL_COL and c in df_target.columns]

    # --- "before": untouched data through Member 2's pipeline ---------------
    results_before = _cross_eval(df_source, df_target, source_name, target_name)

    # --- standardise in source space, then CORAL ----------------------------
    scaler = StandardScaler().fit(df_source[feats])
    Xs = scaler.transform(df_source[feats])
    Xt = scaler.transform(df_target[feats])
    Xs_aligned = coral_align(Xs, Xt, lambda_reg=lambda_reg, match_mean=True)

    src_scaled = pd.DataFrame(Xs, columns=feats, index=df_source.index)
    src_aligned = pd.DataFrame(Xs_aligned, columns=feats, index=df_source.index)
    tgt_scaled = pd.DataFrame(Xt, columns=feats, index=df_target.index)
    src_aligned[LABEL_COL] = df_source[LABEL_COL].values
    tgt_scaled_l = tgt_scaled.copy()
    tgt_scaled_l[LABEL_COL] = df_target[LABEL_COL].values

    # --- "after": aligned source vs. target ---------------------------------
    results_after = _cross_eval(src_aligned, tgt_scaled_l, source_name, target_name)

    # --- divergence before / after (same standardised space) ----------------
    div_before = divergence_report(src_scaled, tgt_scaled, source_name, target_name)
    div_after = divergence_report(src_aligned[feats], tgt_scaled, source_name, target_name)

    cov_s, cov_a, cov_t = (np.cov(M, rowvar=False) for M in (Xs, Xs_aligned, Xt))
    return {
        "results_before": results_before,
        "results_after": results_after,
        "divergence_before": div_before,
        "divergence_after": div_after,
        "covariance_shift": {
            "before": float(np.linalg.norm(cov_s - cov_t)),
            "after": float(np.linalg.norm(cov_a - cov_t)),
            "cov_source": cov_s, "cov_aligned": cov_a, "cov_target": cov_t,
            "features": feats,
        },
        "aligned_source": src_aligned,
    }
