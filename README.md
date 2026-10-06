# Databyte

Cross-dataset intrusion detection research project using CICIDS2017 and UNSW-NB15.

## Current implementation

- `src/config.py`: project paths, random seed, semantic feature names, and model hyperparameters.
- `src/preprocessing.py`: chunked CSV cleaning, binary labels, balanced sampling, and Parquet export.
- `tests/mock_data_generator.py`: small synthetic CICIDS2017 and UNSW-NB15 CSVs.
- `tests/test_preprocessing.py`: preprocessing checks for both mock datasets.
- `requirements.txt`: pinned Python dependencies.

## Next implementation step

Implement `src/feature_mapping.py` to reconcile the shared CICIDS2017 and UNSW-NB15 behavioral features into a common canonical schema. Add tests confirming both datasets produce the same ordered feature columns.

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
