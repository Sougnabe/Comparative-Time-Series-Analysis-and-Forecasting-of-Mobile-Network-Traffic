r"""
Train and evaluate forecasting models on selected squares.

The workspace data covers Oct-Nov 2013, so the requested Dec 16-22 holdout is not
available. The script therefore uses the last available week as test.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PARQUET_DIR = PROJECT_ROOT / 'data' / 'processed' / 'parquet'
OUT_DIR = PROJECT_ROOT / 'artifacts' / 'models'
OUT_DIR.mkdir(parents=True, exist_ok=True)

LAGS = [1, 2, 3, 6, 12, 72, 144]
ROLL_WINDOWS = [6, 12, 72]


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred):
    denom = np.where(np.abs(y_true) < 1e-8, 1e-8, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def load_square_series(square_id):
    path = PARQUET_DIR / f'square_{int(square_id)}.parquet'
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_parquet(path)
    for col in ['Square id', 'Time Interval', 'CDR']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['Time Interval', 'CDR'])
    df['Time Interval'] = df['Time Interval'].astype('int64')
    df['CDR'] = df['CDR'].astype('float64')

    series = df.groupby('Time Interval')['CDR'].sum().sort_index()
    index = pd.to_datetime(series.index, unit='ms', utc=True).tz_convert(None)
    series = pd.Series(series.values, index=index, name='CDR')
    full_index = pd.date_range(series.index.min(), series.index.max(), freq='10min')
    return series.reindex(full_index).fillna(0.0)


def add_time_features(index):
    minutes = index.hour * 60 + index.minute
    return np.column_stack([
        index.hour.astype(float),
        index.dayofweek.astype(float),
        (index.dayofweek >= 5).astype(float),
        minutes.astype(float),
        np.sin(2 * np.pi * minutes / (24 * 60)),
        np.cos(2 * np.pi * minutes / (24 * 60)),
        np.sin(2 * np.pi * index.dayofweek / 7.0),
        np.cos(2 * np.pi * index.dayofweek / 7.0),
        index.is_month_end.astype(float),
    ])


def build_sample(values, index, i):
    logged = np.log1p(np.maximum(values, 0.0))
    sample = []
    for lag in LAGS:
        sample.append(logged[i - lag])
    for window in ROLL_WINDOWS:
        window_values = logged[i - window:i]
        sample.extend([
            window_values.mean(),
            window_values.std(ddof=0),
            window_values.min(),
            window_values.max(),
        ])
    sample.extend(add_time_features(index[i:i + 1])[0].tolist())
    return np.asarray(sample, dtype=float)


def make_training_matrix(series):
    values = series.values.astype(float)
    index = series.index
    start = max(LAGS + ROLL_WINDOWS)
    rows = []
    targets = []
    for i in range(start, len(series)):
        rows.append(build_sample(values, index, i))
        targets.append(np.log1p(max(values[i], 0.0)))
    return np.asarray(rows, dtype=float), np.asarray(targets, dtype=float)


def standardize_fit(X):
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale == 0] = 1.0
    return mean, scale


def standardize_transform(X, mean, scale):
    return (X - mean) / scale


def fit_ridge(X, y, alpha=1.0):
    ones = np.ones((X.shape[0], 1), dtype=float)
    Xb = np.hstack([ones, X])
    reg = alpha * np.eye(Xb.shape[1], dtype=float)
    reg[0, 0] = 0.0
    coef = np.linalg.solve(Xb.T @ Xb + reg, Xb.T @ y)
    return coef


def predict_ridge(coef, X):
    ones = np.ones((X.shape[0], 1), dtype=float)
    raw = np.hstack([ones, X]) @ coef
    return np.expm1(np.clip(raw, 0.0, 12.0))


def fit_mlp(X, y, hidden=32, epochs=250, lr=0.01, l2=1e-4, seed=42):
    rng = np.random.default_rng(seed)
    n_samples, n_features = X.shape
    w1 = rng.normal(0.0, 0.05, size=(n_features, hidden))
    b1 = np.zeros(hidden, dtype=float)
    w2 = rng.normal(0.0, 0.05, size=(hidden, 1))
    b2 = np.zeros(1, dtype=float)

    for _ in range(epochs):
        z1 = X @ w1 + b1
        a1 = np.tanh(z1)
        yhat = (a1 @ w2 + b2).reshape(-1)
        error = yhat - y

        grad_y = (2.0 / n_samples) * error
        grad_w2 = a1.T @ grad_y[:, None] + l2 * w2
        grad_b2 = np.array([grad_y.sum()])
        grad_a1 = grad_y[:, None] @ w2.T
        grad_z1 = grad_a1 * (1.0 - np.tanh(z1) ** 2)
        grad_w1 = X.T @ grad_z1 + l2 * w1
        grad_b1 = grad_z1.sum(axis=0)

        for grad in (grad_w2, grad_b2, grad_w1, grad_b1):
            np.clip(grad, -5.0, 5.0, out=grad)

        w2 -= lr * grad_w2
        b2 -= lr * grad_b2
        w1 -= lr * grad_w1
        b1 -= lr * grad_b1

    return {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}


def predict_mlp(model, X):
    z1 = X @ model['w1'] + model['b1']
    a1 = np.tanh(z1)
    raw = (a1 @ model['w2'] + model['b2']).reshape(-1)
    return np.expm1(np.clip(raw, 0.0, 12.0))


def train_test_split(series, test_start=None, test_end=None):
    if test_start and test_end:
        t0 = pd.to_datetime(test_start)
        t1 = pd.to_datetime(test_end)
        if t1 <= series.index.max() and t0 >= series.index.min():
            test = series[t0:t1]
            train = series[:t0 - pd.Timedelta(minutes=10)]
            return train, test
        print('Requested test range not in data; falling back to last available week.')

    t1 = series.index.max()
    t0 = t1 - pd.Timedelta(days=7) + pd.Timedelta(minutes=10)
    test = series[t0:t1]
    train = series[:t0 - pd.Timedelta(minutes=10)]
    return train, test


# Ridge recursive forecasting removed. We use MLP recursive forecast instead.


def recursive_forecast_mlp(model, history_values, history_index, forecast_index, mean, scale):
    buffer_values = list(history_values)
    local_index = pd.DatetimeIndex(history_index)
    preds = []
    for timestamp in forecast_index:
        local_index = local_index.append(pd.DatetimeIndex([timestamp]))
        sample = build_sample(np.asarray(buffer_values, dtype=float), local_index, len(buffer_values))
        sample = standardize_transform(sample[None, :], mean, scale)
        pred = float(predict_mlp(model, sample)[0])
        preds.append(pred)
        buffer_values.append(pred)
    return np.asarray(preds, dtype=float)


def recursive_forecast_ridge(coef, history_values, history_index, forecast_index, mean, scale):
    buffer_values = list(history_values)
    local_index = pd.DatetimeIndex(history_index)
    preds = []
    for timestamp in forecast_index:
        local_index = local_index.append(pd.DatetimeIndex([timestamp]))
        sample = build_sample(np.asarray(buffer_values, dtype=float), local_index, len(buffer_values))
        sample = standardize_transform(sample[None, :], mean, scale)
        pred = float(predict_ridge(coef, sample)[0])
        preds.append(pred)
        buffer_values.append(pred)
    return np.asarray(preds, dtype=float)




def load_totals():
    totals_path = PROJECT_ROOT / 'artifacts' / 'per_square_totals.csv'
    if totals_path.exists():
        totals = pd.read_csv(totals_path)
    else:
        records = []
        if not PARQUET_DIR.exists():
            raise FileNotFoundError(PARQUET_DIR)
        for path in sorted(PARQUET_DIR.glob('square_*.parquet')):
            square_id = int(path.stem.replace('square_', ''))
            df = pd.read_parquet(path, columns=['CDR'])
            cdr = pd.to_numeric(df['CDR'], errors='coerce').fillna(0.0)
            records.append({'square_id': square_id, 'total_cdr': float(cdr.sum())})
        totals = pd.DataFrame.from_records(records)
        if totals.empty:
            raise RuntimeError('No Parquet square files found to build totals.')
        totals_path.parent.mkdir(parents=True, exist_ok=True)
        totals.to_csv(totals_path, index=False)

    totals['square_id'] = totals['square_id'].astype(int)
    totals['total_cdr'] = pd.to_numeric(totals['total_cdr'], errors='coerce')
    return totals.dropna(subset=['total_cdr'])


def evaluate_models_for_square(square_id, test_start=None, test_end=None):
    series = load_square_series(square_id)
    train, test = train_test_split(series, test_start, test_end)

    X_train, y_train = make_training_matrix(train)
    mean, scale = standardize_fit(X_train)
    X_train_s = standardize_transform(X_train, mean, scale)

    results = []

    t0 = time.time()
    persistence = np.repeat(train.iloc[-1], len(test))
    results.append({
        'square': square_id,
        'model': 'persistence',
        'train_time': 0.0,
        'exec_time': time.time() - t0,
        'MAE': mae(test.values, persistence),
        'MAPE': mape(test.values, persistence),
        'RMSE': rmse(test.values, persistence),
    })
    # Ridge model added back into evaluation flow.
    t0 = time.time()
    ridge_coef = fit_ridge(X_train_s, y_train, alpha=10.0)
    ridge_train_time = time.time() - t0
    ridge_pred = recursive_forecast_ridge(ridge_coef, train.values, train.index, test.index, mean, scale)
    results.append({
        'square': square_id,
        'model': 'ridge_seasonal',
        'train_time': ridge_train_time,
        'exec_time': 0.0,
        'MAE': mae(test.values, ridge_pred),
        'MAPE': mape(test.values, ridge_pred),
        'RMSE': rmse(test.values, ridge_pred),
    })

    t0 = time.time()
    mlp_model = fit_mlp(X_train_s, y_train, hidden=24, epochs=250, lr=0.0015, l2=5e-4)
    mlp_train_time = time.time() - t0
    mlp_pred = recursive_forecast_mlp(mlp_model, train.values, train.index, test.index, mean, scale)
    results.append({
        'square': square_id,
        'model': 'mlp_seasonal',
        'train_time': mlp_train_time,
        'exec_time': 0.0,
        'MAE': mae(test.values, mlp_pred),
        'MAPE': mape(test.values, mlp_pred),
        'RMSE': rmse(test.values, mlp_pred),
    })

    predictions = pd.DataFrame({
        'timestamp': test.index,
        'actual': test.values,
        'pred_persistence': persistence,
        'pred_ridge_seasonal': ridge_pred,
        'pred_mlp_seasonal': mlp_pred,
    })
    predictions.to_csv(OUT_DIR / f'square_{square_id}_test_predictions.csv', index=False)
    return results


def main(parsed_args):
    totals = load_totals()
    top_square = int(totals.sort_values('total_cdr', ascending=False).iloc[0]['square_id'])
    squares = parsed_args.squares or [top_square, 4159, 4556]
    print('Squares to evaluate:', squares)

    all_results = []
    for square_id in squares:
        print('Processing square', square_id)
        all_results.extend(evaluate_models_for_square(square_id, parsed_args.test_start, parsed_args.test_end))

    dfres = pd.DataFrame(all_results)
    dfres.to_csv(OUT_DIR / 'models_performance.csv', index=False)
    print('Done. Results in', OUT_DIR)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--squares', type=int, nargs='*', help='square ids to evaluate')
    parser.add_argument('--test-start', type=str, default=None, help='test start timestamp (YYYY-MM-DD)')
    parser.add_argument('--test-end', type=str, default=None, help='test end timestamp (YYYY-MM-DD)')
    cli_args = parser.parse_args()
    main(cli_args)
