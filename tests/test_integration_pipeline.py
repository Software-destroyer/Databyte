"""Integration checks for the canonical data-to-report workflow."""
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import MODEL_HYPERPARAMETERS
from src.domain_alignment import align_bidirectional, coral_align
from src.evaluation import consolidate_results, compute_grr_table
from src.feature_mapping import FeatureMapper
from src.preprocessing import DataPreprocessor, UNSW_NB15_COLUMNS
from tests.mock_data_generator import generate_mock_data


def test_full_synthetic_workflow(tmp_path: Path) -> None:
    raw = generate_mock_data(tmp_path / "raw", rows=160, seed=17)
    processor = DataPreprocessor(chunk_size=31, sample_size=160, output_dir=tmp_path / "processed")
    clean_cic = processor.process_cicids2017(raw["cicids2017"])
    clean_unsw = processor.process_unsw_nb15(raw["unsw_nb15"])
    mapper = FeatureMapper(output_dir=tmp_path / "processed")
    cic = pd.read_parquet(mapper.process_file(clean_cic, dataset="cicids2017"))
    unsw = pd.read_parquet(mapper.process_file(clean_unsw, dataset="unsw_nb15"))
    assert list(cic.columns) == list(unsw.columns)
    assert np.allclose(cic["flow_duration"].median(), pd.read_parquet(clean_cic)["Flow Duration"].median() / 1_000_000)

    from src.baseline_models import evaluate_holdout_models
    from src.cross_dataset import evaluate_bidirectional
    same = pd.concat([
        evaluate_holdout_models(cic, "cicids2017", models=["decision_tree"]),
        evaluate_holdout_models(unsw, "unsw_nb15", models=["decision_tree"]),
    ], ignore_index=True)
    raw_transfer = evaluate_bidirectional(cic, unsw, models=["decision_tree"])
    aligned, _ = align_bidirectional(cic, unsw, models=["decision_tree"])
    combined = consolidate_results(same, raw_transfer, aligned)
    assert len(combined) == 6
    assert len(compute_grr_table(combined)) == 2


def test_headerless_official_unsw_directory(tmp_path: Path) -> None:
    raw_dir = tmp_path / "unsw"
    raw_dir.mkdir()
    rng = np.random.default_rng(3)
    for part in (1, 2):
        frame = pd.DataFrame({col: np.arange(20, dtype=np.float64) + part for col in UNSW_NB15_COLUMNS})
        frame["proto"] = "tcp"
        frame["state"] = "FIN"
        frame["service"] = "http"
        frame["attack_cat"] = np.where(np.arange(20) % 2, "Generic", "Normal")
        frame["label"] = np.arange(20) % 2
        frame.to_csv(raw_dir / f"UNSW-NB15_{part}.csv", index=False, header=False)
    # This metadata table must not be mixed into the flow table.
    pd.DataFrame({"Name": ["dur"], "Type": ["float"]}).to_csv(raw_dir / "UNSW-NB15_features.csv", index=False)
    out = DataPreprocessor(sample_size=40, output_dir=tmp_path / "processed").process_unsw_nb15(raw_dir)
    result = pd.read_parquet(out)
    assert len(result) == 40
    assert result["label"].value_counts().to_dict() == {0: 20, 1: 20}


def test_requested_classifier_configuration() -> None:
    assert MODEL_HYPERPARAMETERS["random_forest"]["n_estimators"] == 100
    assert MODEL_HYPERPARAMETERS["svm_linear"]["kernel"] == "linear"
    assert MODEL_HYPERPARAMETERS["svm_rbf"]["kernel"] == "rbf"
    assert MODEL_HYPERPARAMETERS["xgboost"]["max_depth"] == 6
    assert MODEL_HYPERPARAMETERS["xgboost"]["learning_rate"] == 0.1


def test_coral_matches_dimensions() -> None:
    source = np.arange(60, dtype=float).reshape(20, 3)
    target = np.arange(90, dtype=float).reshape(30, 3) * 2
    assert coral_align(source, target).shape == source.shape
