**Failure Analysis & Diagnostic Plan**

Overview
--------
This brief document outlines steps to analyze forecasting failures observed in model outputs, and provides quick reproducible checks.

Immediate checks
----------------
- Compare residual distributions (error = actual - pred) across models and time-of-day.
- Identify top 1% largest absolute errors and inspect timestamps for anomalies (holidays, outages).
- Plot rolling MAE over time (window = 144 intervals = 1 day) to locate drift periods.

Suggested tools / scripts
------------------------
- `scripts/run_experiments.py` — runs small sweeps and produces `artifacts/experiments/experiments_summary.csv`.
- `scripts/failure_analysis.py` — (not present) recommended: automate residual extraction and generate a short notebook-friendly CSV for plotting.

Next steps
----------
1. Run `scripts/run_experiments.py` to collect variant performances.
2. Use `artifacts/experiments/experiments_summary.csv` to identify hyperparams that reduce extreme errors.
3. Create targeted retraining for squares/times showing large errors and add corrective feature engineering.
