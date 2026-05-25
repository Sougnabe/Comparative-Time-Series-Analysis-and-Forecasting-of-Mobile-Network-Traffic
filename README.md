Project: Mobile Traffic Forecasting (Milan)

This repository contains a reproducible Python pipeline for:
- converting Telecom Italia TXT files into Parquet files partitioned by grid square,
- exploratory data analysis (EDA),
- training and evaluating forecasting models,
- generating diagnostic plots for model inspection.

Repository layout:
- `data/raw/`: source data (TXT and geojson). These raw files are large and not included in the release.
- `data/processed/parquet/`: processed Parquet files per square (`square_<id>.parquet`).
- `src/`: core modules (loader, conversion utilities).
- `scripts/`: pipeline entry scripts and helpers.
- `artifacts/eda/`: EDA outputs (plots, summary JSON).
- `artifacts/models/`: model predictions and metrics.

Quick start (Windows PowerShell):

1. Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Convert raw TXT files to Parquet (sample run may be faster):

```powershell
python src\convert_all.py
```

3. Run exploratory data analysis:

```powershell
python scripts\run_eda.py
```

4. Train and evaluate models:

```powershell
python scripts\train_models.py
```

5. Generate model diagnostic plots:

```powershell
python scripts\plot_model_diagnostics.py
```

Note: If `artifacts/per_square_totals.csv` is missing, it will be rebuilt automatically from the Parquet files.
