# Databyte

Cross-dataset intrusion detection research project using CICIDS2017 and UNSW-NB15.

## Current implementation

- `src/config.py`: project paths, random seed, semantic feature names, and model hyperparameters.
- `src/preprocessing.py`: chunked CSV cleaning, binary labels, balanced sampling, and Parquet export.
- `src/evaluation.py`: consolidated evaluation, GRR computation, ablation summaries, and metric reporting (Module 4).
- `dashboard/app.py`: Streamlit/Plotly SOC-style dashboard for interactive inspection of results (Module 4).
- `main.py`: CLI orchestration for mock/small workflows and full pipeline steps (Module 4).
- `tests/mock_data_generator.py`: small synthetic CICIDS2017 and UNSW-NB15 CSVs.
- `tests/test_preprocessing.py`: preprocessing checks for both mock datasets.
- `tests/test_evaluation.py`: evaluation consolidation, GRR, and ablation tests (Module 4).
- `tests/test_cli.py`: CLI argument parsing and error handling tests (Module 4).
- `requirements.txt`: pinned Python dependencies.

## Next implementation step

Implement `src/feature_mapping.py` to reconcile the shared CICIDS2017 and UNSW-NB15 behavioral features into a common canonical schema. Add tests confirming both datasets produce the same ordered feature columns.

## CLI usage (Module 4)

```powershell
# Quick demo with mock data (no datasets needed)
python main.py mock

# Launch the interactive dashboard
python main.py dashboard

# Generate mock CSV fixtures
python main.py generate-mock --rows 500

# Full pipeline (requires real CSV files)
python main.py run --cicids data/raw/cicids2017/file.csv --unsw data/raw/unsw_nb15/file.csv

# Individual pipeline steps
python main.py preprocess --cicids <csv> --unsw <csv>
python main.py baselines [--models decision_tree random_forest]
python main.py transfer  [--models decision_tree random_forest]
python main.py evaluate  [--results-csv results/consolidated_results.csv]
```

## Module ownership

1. **Pynskhemlang Marbaniang — ingestion, cleaning, and semantic mapping:** preprocessing, canonical feature mapping, and related tests.
2. **Tehjib Almas Junaid — ML baselines and transfer:** same-dataset model evaluation and bidirectional zero-shot cross-dataset evaluation.
3. **Bamelaai Marbaniang — divergence and alignment:** Wasserstein/KL/MMD analysis, feature selection/scaling, and unsupervised CORAL.
4. **Md Sabique Huda — evaluation and application:** metrics, GRR/ablations, Streamlit dashboard, Plotly visualizations, and CLI orchestration.

## Generate mock input data

```powershell
python tests/mock_data_generator.py
```

This writes files beneath `data/raw/cicids2017/` and `data/raw/unsw_nb15/`.

## Run tests

```powershell
python -m pip install -r requirements.txt
python -m pytest
```

Raw/processed data, trained models, generated results, and Python caches are ignored by Git. Reference presentations and research documents are kept in `docs/`.
