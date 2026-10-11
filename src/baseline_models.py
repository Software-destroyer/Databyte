"""Same-dataset baselines for binary network intrusion detection.

The public functions accept a DataFrame whose features have already been
harmonized. Numeric imputation and scaling are inside sklearn Pipelines, so
those transformations are fitted only on the training partition.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from joblib import dump

from src.config import MODEL_HYPERPARAMETERS, MODELS_DIR, RANDOM_SEED

DEFAULT_MODELS = (
    "decision_tree",
    "random_forest",
    "logistic_regression",
    "knn",
    "svm",
)
MODEL_DISPLAY_NAMES = {
    "random_forest": "RF",
    "decision_tree": "DT",
    "logistic_regression": "LR",
    "knn": "KNN",
    "svm": "SVM",
}
MODEL_KEYS_BY_ALIAS = {
    **{key: key for key in MODEL_DISPLAY_NAMES},
    **{value.casefold(): key for key, value in MODEL_DISPLAY_NAMES.items()},
}
LABEL_COLUMN_CANDIDATES = ("label", "class", "attack_cat")
METRIC_COLUMNS = (
    "accuracy",
    "precision",
    "recall",
    "macro_f1",
    "mcc",
    "far",
    "roc_auc",
    "pr_auc",
)
RESULT_COLUMNS = (
    "dataset",
    "training_domain",
    "test_domain",
    "evaluation_mode",
    "model",
    "n_train",
    "n_test",
    "train_benign",
    "train_attack",
    "test_benign",
    "test_attack",
    *METRIC_COLUMNS,
)


def resolve_label_column(frame: pd.DataFrame, label_column: str | None = None) -> str:
    """Resolve the label column without depending on source capitalization."""
    if label_column is not None:
        matches = [column for column in frame.columns if str(column).casefold() == label_column.casefold()]
        if len(matches) != 1:
            raise ValueError(f"Label column {label_column!r} is missing or ambiguous")
        return matches[0]
    lookup = {str(column).strip().casefold(): column for column in frame.columns}
    for candidate in LABEL_COLUMN_CANDIDATES:
        if candidate in lookup:
            return lookup[candidate]
    raise ValueError("DataFrame must contain a binary 'label' column")


def encode_binary_labels(labels: pd.Series) -> np.ndarray:
    """Convert common benign/normal and binary labels to int8 values."""
    if labels.isna().any():
        raise ValueError("Labels contain missing values")
    numeric = pd.to_numeric(labels, errors="coerce")
    if numeric.notna().all() and numeric.isin([0, 1]).all():
        encoded = numeric.to_numpy(dtype=np.int8)
    else:
        text = labels.astype(str).str.strip().str.casefold()
        encoded = (~text.isin({"benign", "normal", "0", "false", "background"})).to_numpy(dtype=np.int8)
    if not np.isin(encoded, [0, 1]).all():
        raise ValueError("Labels must map to binary values 0 (benign) and 1 (attack)")
    if np.unique(encoded).size != 2:
        raise ValueError("Both benign/normal and attack labels are required")
    return encoded


def split_features_and_labels(
    frame: pd.DataFrame,
    label_column: str | None = None,
    feature_columns: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, np.ndarray, str]:
    """Return numeric features, binary labels, and the resolved label name.

    Feature columns must already be semantically harmonized when comparing
    different datasets. Unparseable non-missing feature values are rejected
    instead of silently turning entire categorical fields into missing data.
    """
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Input must be a non-empty pandas DataFrame")
    if frame.columns.has_duplicates:
        raise ValueError("DataFrame contains duplicate column names")
    resolved_label = resolve_label_column(frame, label_column)
    selected = list(feature_columns) if feature_columns is not None else [
        column for column in frame.columns if column != resolved_label
    ]
    if not selected:
        raise ValueError("No feature columns are available")
    missing = [column for column in selected if column not in frame.columns]
    if missing:
        raise ValueError(f"Requested feature columns are missing: {missing}")
    if resolved_label in selected:
        raise ValueError("The label column cannot be used as a model feature")

    features = frame.loc[:, selected].copy()
    for column in selected:
        original = features[column]
        converted = pd.to_numeric(original, errors="coerce")
        invalid = original.notna() & converted.isna()
        if invalid.any():
            examples = original.loc[invalid].astype(str).drop_duplicates().head(3).tolist()
            raise ValueError(f"Feature {column!r} contains non-numeric values, e.g. {examples}")
        features[column] = converted.replace([np.inf, -np.inf], np.nan).astype("float64")

    all_missing = [column for column in features if features[column].isna().all()]
    if all_missing:
        raise ValueError(f"Features contain no finite observations: {all_missing}")
    labels = encode_binary_labels(frame[resolved_label])
    return features, labels, resolved_label


def build_estimator(model_name: str):
    """Construct one configured classifier by its stable config key."""
    key = MODEL_KEYS_BY_ALIAS.get(model_name.strip().casefold(), model_name.strip().casefold())
    if key not in MODEL_HYPERPARAMETERS:
        raise ValueError(f"Unknown model {model_name!r}; available models: {', '.join(DEFAULT_MODELS)}")
    params = deepcopy(MODEL_HYPERPARAMETERS[key])
    if key == "decision_tree":
        return DecisionTreeClassifier(**params)
    if key == "random_forest":
        return RandomForestClassifier(**params)
    if key == "logistic_regression":
        return LogisticRegression(**params)
    if key == "knn":
        return KNeighborsClassifier(**params)
    if key == "svm":
        return SVC(**params)
    raise AssertionError("Model configuration and factory are inconsistent")


def build_pipeline(model_name: str) -> Pipeline:
    """Build a leakage-safe imputation/scaling/classifier pipeline."""
    steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    key = MODEL_KEYS_BY_ALIAS.get(model_name.strip().casefold(), model_name.strip().casefold())
    if key in {"logistic_regression", "knn", "svm"}:
        steps.append(("scaler", StandardScaler()))
    steps.append(("classifier", build_estimator(key)))
    return Pipeline(steps)


def classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    scores: Sequence[float] | None = None,
) -> dict[str, float | None]:
    """Calculate NIDS metrics; undefined binary ranking/FAR metrics are None."""
    truth = np.asarray(y_true, dtype=np.int8)
    predicted = np.asarray(y_pred, dtype=np.int8)
    if truth.ndim != 1 or predicted.shape != truth.shape:
        raise ValueError("y_true and y_pred must be one-dimensional arrays of equal length")
    if not np.isin(truth, [0, 1]).all() or not np.isin(predicted, [0, 1]).all():
        raise ValueError("Metrics require binary labels encoded as 0 and 1")
    tn, fp, fn, tp = confusion_matrix(truth, predicted, labels=[0, 1]).ravel()
    far_denominator = int(fp + tn)
    result: dict[str, float | None] = {
        "accuracy": float(accuracy_score(truth, predicted)),
        "precision": float(precision_score(truth, predicted, zero_division=0)),
        "recall": float(recall_score(truth, predicted, zero_division=0)),
        "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(truth, predicted)),
        "far": float(fp / far_denominator) if far_denominator else None,
        "roc_auc": None,
        "pr_auc": None,
    }
    if scores is not None and np.unique(truth).size == 2:
        score_values = np.asarray(scores, dtype=np.float64)
        if score_values.shape != truth.shape or not np.isfinite(score_values).all():
            raise ValueError("Scores must be finite and have the same length as y_true")
        result["roc_auc"] = float(roc_auc_score(truth, score_values))
        result["pr_auc"] = float(average_precision_score(truth, score_values))
    return result


def positive_class_scores(estimator: Pipeline, features: pd.DataFrame) -> np.ndarray | None:
    """Return positive-class probabilities or decision scores if available."""
    if hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(features)
        classifier = estimator.named_steps["classifier"] if isinstance(estimator, Pipeline) else estimator
        classes = classifier.classes_
        positive_index = int(np.flatnonzero(classes == 1)[0])
        return np.asarray(probabilities[:, positive_index], dtype=np.float64)
    if hasattr(estimator, "decision_function"):
        return np.asarray(estimator.decision_function(features), dtype=np.float64)
    return None


def class_counts(labels: Sequence[int]) -> tuple[int, int]:
    """Return (benign_count, attack_count)."""
    values = np.asarray(labels, dtype=np.int8)
    return int(np.count_nonzero(values == 0)), int(np.count_nonzero(values == 1))


def evaluate_holdout_models(
    frame: pd.DataFrame,
    dataset: str,
    models: Iterable[str] | None = None,
    label_column: str | None = None,
    feature_columns: Sequence[str] | None = None,
    test_size: float = 0.2,
    random_state: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Run stratified same-dataset holdout baselines and return stable rows."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    features, labels, _ = split_features_and_labels(frame, label_column, feature_columns)
    counts = np.bincount(labels, minlength=2)
    if counts.min() < 2:
        raise ValueError("Each class needs at least two rows for a stratified holdout")
    indices = np.arange(len(labels))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
    selected_models = tuple(models) if models is not None else DEFAULT_MODELS
    if not selected_models:
        raise ValueError("At least one model must be selected")
    train_benign, train_attack = class_counts(labels[train_idx])
    test_benign, test_attack = class_counts(labels[test_idx])
    rows = []
    for model_name in selected_models:
        estimator = build_pipeline(model_name)
        estimator.fit(features.iloc[train_idx], labels[train_idx])
        predictions = estimator.predict(features.iloc[test_idx]).astype(np.int8)
        scores = positive_class_scores(estimator, features.iloc[test_idx])
        row = {
            "dataset": dataset,
            "training_domain": dataset,
            "test_domain": dataset,
            "evaluation_mode": "same_dataset_holdout",
            "model": model_name,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "train_benign": train_benign,
            "train_attack": train_attack,
            "test_benign": test_benign,
            "test_attack": test_attack,
            **classification_metrics(labels[test_idx], predictions, scores),
        }
        rows.append(row)
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def evaluate_dataset(
    dataset: str,
    data: pd.DataFrame,
    **kwargs,
) -> pd.DataFrame:
    """Convenience alias for the same-dataset baseline evaluation."""
    return evaluate_holdout_models(data, dataset=dataset, **kwargs)


def get_models() -> dict[str, object]:
    """Return fresh configured instances of the five project classifiers."""
    return {display: build_estimator(key) for key, display in MODEL_DISPLAY_NAMES.items()}


def train_and_evaluate(
    X_train,
    y_train,
    X_test,
    y_test,
    model_name: str,
    model,
) -> dict:
    """Fit one classifier and return the metrics required by the project brief."""
    train_labels = encode_binary_labels(pd.Series(y_train))
    test_labels = encode_binary_labels(pd.Series(y_test))
    model.fit(X_train, train_labels)
    predictions = np.asarray(model.predict(X_test), dtype=np.int8)
    metrics = classification_metrics(
        test_labels, predictions, positive_class_scores(model, X_test)
        if isinstance(model, Pipeline) else (
            model.predict_proba(X_test)[:, int(np.flatnonzero(model.classes_ == 1)[0])]
            if hasattr(model, "predict_proba") else
            model.decision_function(X_test) if hasattr(model, "decision_function") else None
        )
    )
    return {
        "model_name": model_name,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": float(f1_score(test_labels, predictions, zero_division=0)),
        "mcc": metrics["mcc"],
        "confusion_matrix": confusion_matrix(test_labels, predictions, labels=[0, 1]),
        "classification_report": classification_report(
            test_labels, predictions, labels=[0, 1], target_names=["benign", "attack"],
            zero_division=0,
        ),
    }


def run_same_dataset_evaluation(
    df: pd.DataFrame,
    dataset_name: str,
    test_size: float = 0.2,
) -> pd.DataFrame:
    """Run stratified same-dataset evaluation and persist each scaler/model."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    features, labels, _ = split_features_and_labels(df)
    if np.bincount(labels, minlength=2).min() < 2:
        raise ValueError("Each class needs at least two rows for a stratified split")
    train_idx, test_idx = train_test_split(
        np.arange(len(labels)), test_size=test_size, random_state=RANDOM_SEED,
        stratify=labels,
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(features.iloc[train_idx])
    X_test = scaler.transform(features.iloc[test_idx])
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    safe_dataset_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in dataset_name)
    dump(scaler, MODELS_DIR / f"{safe_dataset_name}_scaler.joblib")
    rows = []
    for model_name, model in get_models().items():
        metrics = train_and_evaluate(
            X_train, labels[train_idx], X_test, labels[test_idx], model_name, model
        )
        dump(model, MODELS_DIR / f"{safe_dataset_name}_{model_name.casefold()}.joblib")
        rows.append({
            "model": model_name,
            "dataset": dataset_name,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "mcc": metrics["mcc"],
        })
    return pd.DataFrame(
        rows, columns=["model", "dataset", "accuracy", "precision", "recall", "f1", "mcc"]
    )
