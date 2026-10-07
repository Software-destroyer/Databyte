"""Zero-shot, bidirectional cross-dataset model transfer evaluation.

The target labels are read only after predictions have been generated. They
are never used to fit or adapt the source-trained classifier.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from src.baseline_models import (
    DEFAULT_MODELS,
    RESULT_COLUMNS,
    build_pipeline,
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
    source_benign, source_attack = class_counts(source_y)
    target_benign, target_attack = class_counts(target_y)
    rows = []
    for model_name in selected_models:
        estimator = build_pipeline(model_name)
        estimator.fit(source_x, source_y)
        predictions = estimator.predict(target_x).astype(np.int8)
        scores = positive_class_scores(estimator, target_x)
        rows.append(
            {
                "dataset": f"{source_domain}_to_{target_domain}",
                "training_domain": source_domain,
                "test_domain": target_domain,
                "evaluation_mode": "zero_shot_cross_dataset",
                "model": model_name,
                "n_train": int(len(source_y)),
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
