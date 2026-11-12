"""Leakage-aware CORAL alignment and aligned cross-dataset evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.baseline_models import (DEFAULT_MODELS, RESULT_COLUMNS, build_estimator,
    class_counts, classification_metrics, positive_class_scores, split_features_and_labels,
    select_training_indices)
from src.config import RANDOM_SEED
from src.divergence import divergence_report

LABEL_COL = "label"


def _matrix_power(matrix: np.ndarray, power: float) -> np.ndarray:
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2.0)
    values = np.maximum(values, 1e-10)
    return (vectors * values ** power) @ vectors.T


def coral_align(X_source: np.ndarray, X_target: np.ndarray, lambda_reg: float = 1e-3,
                match_mean: bool = True) -> np.ndarray:
    """Map source covariance to target covariance using CORAL.

    The target matrix is used without labels. Callers must keep source test
    records out of ``X_source`` to avoid leaking held-out source information.
    """
    source = np.asarray(X_source, dtype=np.float64)
    target = np.asarray(X_target, dtype=np.float64)
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
        raise ValueError("Source and target must be 2-D with matching feature counts")
    if min(len(source), len(target)) < 2:
        raise ValueError("CORAL needs at least two rows in each domain")
    dim = source.shape[1]
    eye = np.eye(dim)
    cov_s = np.atleast_2d(np.cov(source, rowvar=False)) + lambda_reg * eye
    cov_t = np.atleast_2d(np.cov(target, rowvar=False)) + lambda_reg * eye
    aligned = (source - source.mean(axis=0)) @ _matrix_power(cov_s, -0.5) @ _matrix_power(cov_t, 0.5)
    if match_mean:
        aligned = aligned + target.mean(axis=0)
    return aligned


def evaluate_aligned_transfer(source: pd.DataFrame, target: pd.DataFrame,
                              source_domain: str, target_domain: str,
                              models=None, lambda_reg: float = 1e-3,
                              random_state: int = RANDOM_SEED) -> tuple[pd.DataFrame, dict]:
    """Fit source-only models after unsupervised CORAL using target features.

    Source rows are split before imputation, scaling, or CORAL. Target labels
    are only used in the final metric calculation.
    """
    source_x, source_y, _ = split_features_and_labels(source)
    target_x, target_y, _ = split_features_and_labels(target)
    if set(source_x.columns) != set(target_x.columns):
        raise ValueError("Source and target features differ; run FeatureMapper first")
    target_x = target_x.loc[:, source_x.columns]
    if np.bincount(source_y, minlength=2).min() < 2:
        raise ValueError("Each source class needs at least two rows")
    train_idx, _ = train_test_split(np.arange(len(source_y)), test_size=0.2,
                                    stratify=source_y, random_state=random_state)
    x_train, y_train = source_x.iloc[train_idx], source_y[train_idx]
    imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(x_train)
    scaler = StandardScaler().fit(imputer.transform(x_train))
    source_scaled = scaler.transform(imputer.transform(x_train))
    target_scaled = scaler.transform(imputer.transform(target_x))
    aligned = coral_align(source_scaled, target_scaled, lambda_reg=lambda_reg)
    source_view = pd.DataFrame(source_scaled, columns=source_x.columns)
    target_view = pd.DataFrame(target_scaled, columns=source_x.columns)
    aligned_view = pd.DataFrame(aligned, columns=source_x.columns)
    before = divergence_report(source_view, target_view, source_domain, target_domain)
    after = divergence_report(aligned_view, target_view, source_domain, target_domain)
    selected_models = tuple(models) if models is not None else DEFAULT_MODELS
    rows = []
    for model_name in selected_models:
        model_train_idx = select_training_indices(y_train, np.arange(len(y_train)), model_name, random_state)
        model_train_y = y_train[model_train_idx]
        train_benign, train_attack = class_counts(model_train_y)
        estimator = build_estimator(model_name)
        estimator.fit(aligned[model_train_idx], model_train_y)
        predictions = estimator.predict(target_scaled).astype(np.int8)
        scores = positive_class_scores(estimator, target_scaled)
        test_benign, test_attack = class_counts(target_y)
        rows.append({
            "dataset": f"{source_domain}_to_{target_domain}",
            "training_domain": source_domain, "test_domain": target_domain,
            "evaluation_mode": "aligned_cross_dataset", "model": model_name,
            "n_train": len(model_train_y), "n_test": len(target_y),
            "train_benign": train_benign, "train_attack": train_attack,
            "test_benign": test_benign, "test_attack": test_attack,
            **classification_metrics(target_y, predictions, scores),
        })
    result = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    return result, {"divergence_before": before, "divergence_after": after,
                    "aligned_source": aligned_view, "target_features": target_view}


def align_bidirectional(cicids2017: pd.DataFrame, unsw_nb15: pd.DataFrame,
                        models=None, lambda_reg: float = 1e-3,
                        random_state: int = RANDOM_SEED) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Run leakage-aware CORAL transfer in both dataset directions."""
    cic_unsw, cic_detail = evaluate_aligned_transfer(cicids2017, unsw_nb15,
        "cicids2017", "unsw_nb15", models, lambda_reg, random_state)
    unsw_cic, unsw_detail = evaluate_aligned_transfer(unsw_nb15, cicids2017,
        "unsw_nb15", "cicids2017", models, lambda_reg, random_state)
    return pd.concat([cic_unsw, unsw_cic], ignore_index=True), {
        "cicids2017_to_unsw_nb15": cic_detail,
        "unsw_nb15_to_cicids2017": unsw_detail,
    }
