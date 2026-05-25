"""Small experiment sweep for MLP and Ridge hyperparameters.
Saves a CSV to artifacts/experiments/ with MAE/RMSE/MAPE for each run.
"""
from pathlib import Path
import time
import numpy as np
import pandas as pd

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from scripts import train_models as tm

OUT_DIR = PROJECT_ROOT / 'artifacts' / 'experiments'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_hyperparams(square_id, ridge_alphas, mlp_configs):
    series = tm.load_square_series(square_id)
    train, test = tm.train_test_split(series)
    X_train, y_train = tm.make_training_matrix(train)
    mean, scale = tm.standardize_fit(X_train)
    X_train_s = tm.standardize_transform(X_train, mean, scale)

    results = []

    # Ridge sweep
    for alpha in ridge_alphas:
        t0 = time.time()
        coef = tm.fit_ridge(X_train_s, y_train, alpha=alpha)
        train_time = time.time() - t0
        pred = tm.recursive_forecast_ridge(coef, train.values, train.index, test.index, mean, scale)
        results.append({
            'square': square_id,
            'model': 'ridge',
            'alpha': alpha,
            'train_time': train_time,
            'MAE': tm.mae(test.values, pred),
            'RMSE': tm.rmse(test.values, pred),
            'MAPE': tm.mape(test.values, pred),
        })

    # MLP sweep
    for cfg in mlp_configs:
        t0 = time.time()
        model = tm.fit_mlp(X_train_s, y_train, hidden=cfg['hidden'], epochs=cfg['epochs'], lr=cfg['lr'], l2=cfg.get('l2', 1e-4))
        train_time = time.time() - t0
        pred = tm.recursive_forecast_mlp(model, train.values, train.index, test.index, mean, scale)
        results.append({
            'square': square_id,
            'model': 'mlp',
            'hidden': cfg['hidden'],
            'epochs': cfg['epochs'],
            'lr': cfg['lr'],
            'train_time': train_time,
            'MAE': tm.mae(test.values, pred),
            'RMSE': tm.rmse(test.values, pred),
            'MAPE': tm.mape(test.values, pred),
        })

    return results


def main():
    totals = tm.load_totals()
    top_square = int(totals.sort_values('total_cdr', ascending=False).iloc[0]['square_id'])

    ridge_alphas = [0.1, 1.0, 10.0]
    mlp_configs = [
        {'hidden': 16, 'epochs': 50, 'lr': 0.01},
        {'hidden': 32, 'epochs': 100, 'lr': 0.005},
    ]

    all_results = []
    for square in [top_square]:
        print('Running experiments for square', square)
        all_results.extend(evaluate_hyperparams(square, ridge_alphas, mlp_configs))

    df = pd.DataFrame(all_results)
    out_path = OUT_DIR / 'experiments_summary.csv'
    df.to_csv(out_path, index=False)
    print('Experiments saved to', out_path)


if __name__ == '__main__':
    main()
