"""End-to-end tests for mock generation and chunked preprocessing."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import DataPreprocessor
from tests.mock_data_generator import generate_mock_data


@pytest.mark.parametrize("dataset", ["cicids2017", "unsw_nb15"])
def test_mock_csv_preprocesses_to_clean_binary_parquet(tmp_path: Path, dataset: str) -> None:
    paths = generate_mock_data(tmp_path / "raw", rows=250)
    source = paths[dataset]
    processor = DataPreprocessor(
        chunk_size=37,
        sample_size=120,
        output_dir=tmp_path / "processed",
    )

    result = (
        processor.process_cicids2017(source)
        if dataset == "cicids2017"
        else processor.process_unsw_nb15(source)
    )

    assert result.parent == tmp_path / "processed"
    assert result.name == ("cic_cleaned.parquet" if dataset == "cicids2017" else "unsw_cleaned.parquet")
    assert result.is_file()
    cleaned = pd.read_parquet(result)
    assert len(cleaned) == 120
    assert set(cleaned["label"].unique()) == {0, 1}
    assert not cleaned.isna().any().any()
    assert np.isfinite(cleaned.select_dtypes(include=[np.number]).to_numpy()).all()
    assert "constant" not in {column.casefold() for column in cleaned.columns}
    assert len(cleaned.columns) > 1


def test_generator_writes_expected_dataset_columns(tmp_path: Path) -> None:
    paths = generate_mock_data(tmp_path, rows=20)
    cic_columns = set(pd.read_csv(paths["cicids2017"], nrows=0).columns)
    unsw_columns = set(pd.read_csv(paths["unsw_nb15"], nrows=0).columns)
    assert {"Flow Duration", "Total Fwd Packets", "Flow Bytes/s", "Label"} <= cic_columns
    assert {"dur", "spkts", "sbytes", "rate", "label"} <= unsw_columns
