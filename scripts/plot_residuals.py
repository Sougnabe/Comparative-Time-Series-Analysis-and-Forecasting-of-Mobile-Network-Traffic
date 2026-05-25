"""Generate simple residual plots from artifacts/analysis/residuals_summary.csv
Saves PNG histograms to artifacts/analysis/plots
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IN_PATH = PROJECT_ROOT / 'artifacts' / 'analysis' / 'residuals_summary.csv'
OUT_DIR = PROJECT_ROOT / 'artifacts' / 'analysis' / 'plots'
OUT_DIR.mkdir(parents=True, exist_ok=True)

if not IN_PATH.exists():
    print('No residuals summary found at', IN_PATH)
    raise SystemExit(1)

df = pd.read_csv(IN_PATH)

for metric in ['MAE', 'RMSE', 'MAPE']:
    plt.figure(figsize=(6,4))
    df[metric].hist(bins=50)
    plt.title(f'Residuals distribution: {metric}')
    plt.xlabel(metric)
    plt.ylabel('count')
    plt.tight_layout()
    out = OUT_DIR / f'{metric.lower()}_hist.png'
    plt.savefig(out)
    print('Saved', out)
