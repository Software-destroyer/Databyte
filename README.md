# Databyte

Cross-dataset intrusion detection evaluation using CICIDS2017 and UNSW-NB15. The project trains standard classifiers, measures same-dataset performance, tests zero-shot transfer in both directions, and evaluates unsupervised CORAL alignment.

## Pipeline

1. Read one CSV, a directory of CSVs, or a list of CSVs in 50,000-row chunks. UNSW's four raw partitions are preferred when present; metadata and duplicate train/test copies are excluded.
2. Encode benign/normal and attack labels, estimate medians and the 99.9th-percentile infinity replacement from bounded per-feature samples, remove constant numeric columns, and create a class-balanced Parquet sample capped at 200,000 rows.
3. Map both datasets to eight shared numeric flow features. CIC Flow Duration is converted from microseconds to seconds. The official UNSW raw schema has no packet-rate column, so packet rate is derived from directional packet counts divided by duration. Nonmatching categorical features are excluded.
4. Evaluate configured Decision Tree, Random Forest (100 trees), Logistic Regression, linear/RBF SVM, and XGBoost classifiers using stratified holdout baselines and bidirectional zero-shot transfer. Imputation/scaling are fit on source training data. RBF-SVM uses a stratified cap of 10,000 training rows to keep its quadratic kernel cost practical; the recorded `n_train` is the actual number used.
5. Compare Wasserstein, KL, and MMD divergence; run CORAL using source training rows and unlabeled target features; evaluate aligned source models on the target.
6. Consolidate metrics, calculate GRR and deltas, then inspect them in the Streamlit dashboard.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run

Place the extracted CSV files in folders outside Git if preferred. Pass either a dataset folder (the pipeline selects its CSV partitions) or a single CSV file:

```powershell
python main.py run --cicids "D:\datasets\cicids2017\MachineLearningCSV" --unsw "D:\datasets\unsw_nb15\CSV Files"
```

Run individual steps after preprocessing:

```powershell
python main.py preprocess --cicids <CICIDS-folder-or-csv> --unsw <UNSW-folder-or-csv>
python main.py baselines
python main.py transfer
python main.py align
python main.py evaluate
python main.py dashboard
```

Generate miniature synthetic input files and run a demo without downloaded data:

```powershell
python main.py generate-mock --rows 1000
python main.py run --cicids data/raw/cicids2017/cicids2017_mock.csv --unsw data/raw/unsw_nb15/unsw_nb15_mock.csv --models decision_tree
```

The `mock` command creates explicitly marked synthetic dashboard results; never cite those values as experimental findings. Run `python -m pytest` for the test suite.

## Outputs

Cleaned and canonical Parquet files are written to `data/processed/`. Model metrics and divergence tables are written to `results/`. `main.py run` creates same-dataset, raw transfer, aligned transfer, divergence, consolidated, and GRR CSV reports. Raw datasets, generated outputs, models, results, archive review copies, and environments are excluded from Git.

## Research scope

See [docs/IMPLEMENTATION_STATUS.txt](docs/IMPLEMENTATION_STATUS.txt) for the completed real-data run, mean results, and scope notes about older draft materials.


This is an ML evaluation framework/tool that uses established classifier families (Decision Tree, Random Forest, Logistic Regression, SVM, and XGBoost) rather than proposing a new model architecture. The shared feature set and data-selection procedure should be documented in the final report; synthetic smoke-test scores are only software checks.
