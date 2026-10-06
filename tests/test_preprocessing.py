"""Tests for the canonical streaming preprocessing module."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import DataPreprocessor
from tests.mock_data_generator import generate_mock_data


@pytest.mark.parametrize("dataset", ["cicids2017", "unsw_nb15"])
def test_mock_data_is_cleaned_and_exported(tmp_path: Path, dataset: str) -> None:
    inputs = generate_mock_data(tmp_path / "raw", rows=200, seed=42)
    processor = DataPreprocessor(
        chunk_size=50,
        sample_size=100,
        random_state=42,
        output_dir=tmp_path / "processed",
    )

    output = (
        processor.process_cicids2017(inputs[dataset])
        if dataset == "cicids2017"
        else processor.process_unsw_nb15(inputs[dataset])
    )

    expected_name = "cic_cleaned.parquet" if dataset == "cicids2017" else "unsw_cleaned.parquet"
    assert output.name == expected_name
    assert output.is_file()
    cleaned = pd.read_parquet(output)
    assert len(cleaned) == 100
    assert cleaned["label"].value_counts().to_dict() == {0: 50, 1: 50}
    assert not cleaned.isna().any().any()
    assert np.isfinite(cleaned.select_dtypes(include=[np.number]).to_numpy()).all()
    assert "constant" not in {name.casefold() for name in cleaned.columns}


def test_default_chunk_size_is_fifty_thousand() -> None:
    assert DataPreprocessor().chunk_size == 50_000
