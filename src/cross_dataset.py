"""Zero-shot, bidirectional cross-dataset model transfer evaluation.

The target labels are read only after predictions have been generated. They
are never used to fit or adapt the source-trained classifier.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.baseline_models import (
    DEFAULT_MODELS,
    RESULT_COLUMNS,
    build_estimator,
    class_counts,
    classification_metrics,
    positive_class_scores,
    split_features_and_labels,
)
from src.config import RANDOM_SEED


def evaluate_transfer(
    source: pd.DataFrame,
    target: pd.DataFrame,
    source_domain: str,
    target_domain: str,
    models: Iterable[str] | None = None,
    source_label_column: str | None = None,
    target_label_column: str | None = None,
    feature_columns: Sequence[str] | None = None,
    random_state: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Fit on all labeled source rows and evaluate zero-shot on target rows.

    Source and target must already have the same semantically harmonized
    feature names. Target features are reordered to source order; missing or
    extra fields are rejected to prevent accidental schema mismatch.
    """
    source_x, source_y, _ = split_features_and_labels(
        source, source_label_column, feature_columns
    )
    target_x, target_y, _ = split_features_and_labels(
        target, target_label_column, feature_columns
    )
    source_columns = list(source_x.columns)
    missing = [column for column in source_columns if column not in target_x.columns]
    extra = [column for column in target_x.columns if column not in source_columns]
    if missing or extra:
        raise ValueError(
            "Source and target feature schemas differ. Apply semantic feature mapping first. "
            f"Missing in target: {missing}; extra in target: {extra}"
        )
    target_x = target_x.loc[:, source_columns]

    selected_models = tuple(models) if models is not None else DEFAULT_MODELS
    if not selected_models:
        raise ValueError("At least one model must be selected")
    if np.bincount(source_y, minlength=2).min() < 2:
        raise ValueError("Each source class needs at least two rows for a stratified split")
    source_train_idx, _ = train_test_split(
        np.arange(len(source_y)), test_size=0.2, random_state=random_state,
        stratify=source_y,
    )
    source_x_train = source_x.iloc[source_train_idx]
    source_y_train = source_y[source_train_idx]
    source_preprocessor = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    source_x_train_scaled = source_preprocessor.fit_transform(source_x_train)
    target_x_scaled = source_preprocessor.transform(target_x)
    source_benign, source_attack = class_counts(source_y_train)
    target_benign, target_attack = class_counts(target_y)
    rows = []
    for model_name in selected_models:
        estimator = build_estimator(model_name)
        estimator.fit(source_x_train_scaled, source_y_train)
        predictions = estimator.predict(target_x_scaled).astype(np.int8)
        scores = positive_class_scores(estimator, target_x_scaled)
        rows.append(
            {
                "dataset": f"{source_domain}_to_{target_domain}",
                "training_domain": source_domain,
                "test_domain": target_domain,
                "evaluation_mode": "zero_shot_cross_dataset",
                "model": model_name,
                "n_train": int(len(source_y_train)),
                "n_test": int(len(target_y)),
                "train_benign": source_benign,
                "train_attack": source_attack,
                "test_benign": target_benign,
                "test_attack": target_attack,
                **classification_metrics(target_y, predictions, scores),
            }
        )
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def evaluate_bidirectional(
    cicids2017: pd.DataFrame,
    unsw_nb15: pd.DataFrame,
    models: Iterable[str] | None = None,
    cicids_label_column: str | None = None,
    unsw_label_column: str | None = None,
    feature_columns: Sequence[str] | None = None,
    random_state: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Evaluate CICIDS2017→UNSW-NB15 and UNSW-NB15→CICIDS2017."""
    selected_models = tuple(models) if models is not None else DEFAULT_MODELS
    cic_to_unsw = evaluate_transfer(
        source=cicids2017,
        target=unsw_nb15,
        source_domain="cicids2017",
        target_domain="unsw_nb15",
        models=selected_models,
        source_label_column=cicids_label_column,
        target_label_column=unsw_label_column,
        feature_columns=feature_columns,
        random_state=random_state,
    )
    unsw_to_cic = evaluate_transfer(
        source=unsw_nb15,
        target=cicids2017,
        source_domain="unsw_nb15",
        target_domain="cicids2017",
        models=selected_models,
        source_label_column=unsw_label_column,
        target_label_column=cicids_label_column,
        feature_columns=feature_columns,
        random_state=random_state,
    )
    return pd.concat([cic_to_unsw, unsw_to_cic], ignore_index=True).reindex(
        columns=RESULT_COLUMNS
    )


def run_cross_dataset_evaluation(
    df_source: pd.DataFrame,
    df_target: pd.DataFrame,
    source_name: str,
    target_name: str,
) -> pd.DataFrame:
    """Run the project-standard 80%-source to full-target transfer test."""
    result = evaluate_transfer(
        source=df_source,
        target=df_target,
        source_domain=source_name,
        target_domain=target_name,
        models=DEFAULT_MODELS,
    )
    return pd.DataFrame({
        "model": result["model"].map(lambda name: {
            "random_forest": "RF",
            "decision_tree": "DT",
            "logistic_regression": "LR",
            "knn": "KNN",
            "svm": "SVM",
        }[name]),
        "train_dataset": result["training_domain"],
        "test_dataset": result["test_domain"],
        "accuracy": result["accuracy"],
        "precision": result["precision"],
        "recall": result["recall"],
        "f1": [
            (2 * precision * recall / (precision + recall))
            if precision + recall else 0.0
            for precision, recall in zip(result["precision"], result["recall"])
        ],
        "mcc": result["mcc"],
    })


def run_bidirectional_evaluation(
    df_cicids: pd.DataFrame,
    df_unsw: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate CICIDS2017→UNSW-NB15 and UNSW-NB15→CICIDS2017."""
    return pd.concat(
        [
            run_cross_dataset_evaluation(df_cicids, df_unsw, "cicids2017", "unsw_nb15"),
            run_cross_dataset_evaluation(df_unsw, df_cicids, "unsw_nb15", "cicids2017"),
        ],
        ignore_index=True,
    )
