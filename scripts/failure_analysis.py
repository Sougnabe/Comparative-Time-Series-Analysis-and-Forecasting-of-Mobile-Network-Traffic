"""Aggregate residual statistics from per-square predictions.
Writes `artifacts/analysis/residuals_summary.csv` with MAE/RMSE/MAPE per model.
"""
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = PROJECT_ROOT / 'artifacts' / 'models'
OUT_DIR = PROJECT_ROOT / 'artifacts' / 'analysis'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def safe_mape(y, yhat):
    denom = np.where(np.abs(y) < 1e-8, 1e-8, np.abs(y))
    return float(np.mean(np.abs((y - yhat) / denom)) * 100.0)


records = []
for path in sorted(PRED_DIR.glob('*_test_predictions.csv')):
    df = pd.read_csv(path)
    square = int(path.stem.split('_')[1])
    actual = df['actual'].values
    for col in df.columns:
        if col.startswith('pred_'):
            yhat = df[col].values
            mae = float(np.mean(np.abs(actual - yhat)))
            rmse = float(np.sqrt(np.mean((actual - yhat) ** 2)))
            mape = safe_mape(actual, yhat)
            records.append({'square': square, 'model_col': col, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape})

out = pd.DataFrame.from_records(records)
out.to_csv(OUT_DIR / 'residuals_summary.csv', index=False)
print('Residuals summary written to', OUT_DIR / 'residuals_summary.csv')
