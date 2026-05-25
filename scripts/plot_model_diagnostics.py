"""
Generate diagnostic plots for model predictions saved in `artifacts/models`.
Produces: actual vs predicted time series, residuals histogram, predicted vs actual scatter.
Saves PNGs to `artifacts/models/plots/`.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts' / 'models'
PLOT_DIR = OUT / 'plots'
PLOT_DIR.mkdir(parents=True, exist_ok=True)

PERF_CSV = OUT / 'models_performance.csv'

def safe_read_csv(p):
    try:
        return pd.read_csv(p)
    except (FileNotFoundError, PermissionError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        print('Failed to read', p, '->', e)
        return None


def plot_for_square(sq_id):
    preds_f = OUT / f'square_{sq_id}_test_predictions.csv'
    if not preds_f.exists():
        print('Missing predictions for', sq_id)
        return
    df = pd.read_csv(preds_f, parse_dates=['timestamp'])
    df = df.set_index('timestamp')
    actual = df['actual'].astype(float)

    # identify prediction columns
    pred_cols = [c for c in df.columns if c.startswith('pred_') or c.startswith('pred')]

    for pc in pred_cols:
        pred = df[pc].astype(float)
        # timeseries plot (trim long series to last 500 points)
        fig, ax = plt.subplots(figsize=(10,4))
        npoints = min(len(df), 500)
        actual.tail(npoints).plot(ax=ax, label='actual', alpha=0.8)
        pred.tail(npoints).plot(ax=ax, label=pc, alpha=0.8)
        ax.set_title(f'Square {sq_id} — actual vs {pc}')
        ax.legend()
        fig.tight_layout()
        fig.savefig(PLOT_DIR / f'square_{sq_id}_{pc}_timeseries.png')
        plt.close(fig)

        # residuals
        resid = actual - pred
        fig, axes = plt.subplots(1,2,figsize=(10,4))
        sns.histplot(resid.replace([np.inf, -np.inf], np.nan).dropna(), bins=50, ax=axes[0])
        axes[0].set_title(f'Residuals histogram {pc}')
        axes[0].set_xlabel('residual')
        axes[1].plot(resid.replace([np.inf, -np.inf], np.nan).fillna(0).values)
        axes[1].set_title(f'Residuals (time) {pc}')
        fig.tight_layout()
        fig.savefig(PLOT_DIR / f'square_{sq_id}_{pc}_residuals.png')
        plt.close(fig)

        # scatter predicted vs actual
        fig, ax = plt.subplots(figsize=(5,5))
        sns.scatterplot(x=actual, y=pred, alpha=0.3, s=10)
        ax.set_xlabel('actual')
        ax.set_ylabel('predicted')
        ax.set_title(f'Predicted vs Actual {sq_id} {pc}')
        lim = np.nanpercentile(np.concatenate([actual.values, pred.values]), 99)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / f'square_{sq_id}_{pc}_scatter.png')
        plt.close(fig)

    print('Plotted', sq_id)


def main():
    perf = safe_read_csv(PERF_CSV)
    if perf is None:
        print('No performance CSV found at', PERF_CSV)
        return
    squares = sorted(perf['square'].unique())
    for s in squares:
        plot_for_square(int(s))
    print('All plots saved to', PLOT_DIR)

if __name__ == '__main__':
    main()
