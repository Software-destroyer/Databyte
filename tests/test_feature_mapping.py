"""Tests for the semantic feature mapping module in Databyte."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.preprocessing import DataPreprocessor
from src.feature_mapping import FeatureMapper, map_features, get_canonical_features
from tests.mock_data_generator import generate_mock_data


def test_canonical_feature_names_list() -> None:
    features = get_canonical_features(include_label=False)
    assert len(features) >= 9
    assert "flow_duration" in features
    assert "label" not in features

    features_with_label = get_canonical_features(include_label=True)
    assert features_with_label[-1] == "label"
    assert len(features_with_label) == len(features) + 1


def test_both_datasets_produce_identical_ordered_columns(tmp_path: Path) -> None:
    raw_paths = generate_mock_data(tmp_path / "raw", rows=200, seed=42)
    preprocessor = DataPreprocessor(
        chunk_size=50,
        sample_size=100,
        random_state=42,
        output_dir=tmp_path / "processed",
    )

    cic_cleaned = pd.read_parquet(preprocessor.process_cicids2017(raw_paths["cicids2017"]))
    unsw_cleaned = pd.read_parquet(preprocessor.process_unsw_nb15(raw_paths["unsw_nb15"]))

    mapper = FeatureMapper(output_dir=tmp_path / "aligned")
    cic_canonical = mapper.map_dataframe(cic_cleaned, "cicids2017")
    unsw_canonical = mapper.map_dataframe(unsw_cleaned, "unsw_nb15")

    # Column ordering and column names must be identical
    assert list(cic_canonical.columns) == list(unsw_canonical.columns)
    assert list(cic_canonical.columns) == get_canonical_features(include_label=True)

    # Both datasets must have identical sample length and valid values
    assert len(cic_canonical) == 100
    assert len(unsw_canonical) == 100

    # Ensure no NaN or infinite values
    assert not cic_canonical.isna().any().any()
    assert not unsw_canonical.isna().any().any()
    assert np.isfinite(cic_canonical.to_numpy()).all()
    assert np.isfinite(unsw_canonical.to_numpy()).all()

    # Labels must be binary (0 or 1)
    assert set(cic_canonical["label"].unique()) == {0, 1}
    assert set(unsw_canonical["label"].unique()) == {0, 1}


def test_feature_mapper_process_file(tmp_path: Path) -> None:
    raw_paths = generate_mock_data(tmp_path / "raw", rows=200, seed=42)
    preprocessor = DataPreprocessor(
        chunk_size=50,
        sample_size=100,
        random_state=42,
        output_dir=tmp_path / "processed",
    )

    cic_cleaned_path = preprocessor.process_cicids2017(raw_paths["cicids2017"])
    unsw_cleaned_path = preprocessor.process_unsw_nb15(raw_paths["unsw_nb15"])

    mapper = FeatureMapper(output_dir=tmp_path / "aligned")
    cic_aligned_path = mapper.process_file(cic_cleaned_path)
    unsw_aligned_path = mapper.process_file(unsw_cleaned_path)

    assert cic_aligned_path.is_file()
    assert unsw_aligned_path.is_file()
    assert cic_aligned_path.name == "cic_aligned.parquet"
    assert unsw_aligned_path.name == "unsw_aligned.parquet"

    df_cic = pd.read_parquet(cic_aligned_path)
    df_unsw = pd.read_parquet(unsw_aligned_path)
    assert list(df_cic.columns) == list(df_unsw.columns)


def test_unknown_dataset_raises_error() -> None:
    mapper = FeatureMapper()
    dummy_df = pd.DataFrame({"col": [1, 2]})
    with pytest.raises(ValueError, match="Unsupported dataset name"):
        mapper.map_dataframe(dummy_df, "unknown_dataset")
