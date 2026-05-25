import os
import json
import math
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

PARQUET_DIR = os.path.join('data', 'processed', 'parquet')
TOTALS_CSV = os.path.join('artifacts', 'per_square_totals.csv')
OUT_DIR = os.path.join('artifacts', 'eda')
os.makedirs(OUT_DIR, exist_ok=True)

sns.set_theme(style='whitegrid')
plt.rcParams['figure.dpi'] = 140
plt.rcParams['savefig.bbox'] = 'tight'


def load_square_series(square_id):
    path = os.path.join(PARQUET_DIR, f'square_{square_id}.parquet')
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    for col in ['Square id', 'Time Interval', 'CDR']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['Time Interval', 'CDR'])
    df['Time Interval'] = df['Time Interval'].astype('int64')
    df['CDR'] = df['CDR'].astype('float64')
    series = df.groupby('Time Interval')['CDR'].sum().sort_index()
    dt_index = pd.to_datetime(series.index, unit='ms', utc=True).tz_convert(None)
    series = pd.Series(series.values, index=dt_index, name='CDR')
    full_index = pd.date_range(series.index.min(), series.index.max(), freq='10min')
    series = series.reindex(full_index).fillna(0.0)
    series.index.name = 'timestamp'
    return series


def load_totals():
    if os.path.exists(TOTALS_CSV):
        totals = pd.read_csv(TOTALS_CSV)
    else:
        records = []
        if not os.path.isdir(PARQUET_DIR):
            raise FileNotFoundError(PARQUET_DIR)
        for name in sorted(os.listdir(PARQUET_DIR)):
            if not (name.startswith('square_') and name.endswith('.parquet')):
                continue
            square_id = int(name.replace('square_', '').replace('.parquet', ''))
            path = os.path.join(PARQUET_DIR, name)
            df = pd.read_parquet(path, columns=['CDR'])
            cdr = pd.to_numeric(df['CDR'], errors='coerce').fillna(0.0)
            records.append({'square_id': square_id, 'total_cdr': float(cdr.sum())})
        totals = pd.DataFrame.from_records(records)
        if totals.empty:
            raise RuntimeError('No Parquet square files found to build totals.')
        os.makedirs(os.path.dirname(TOTALS_CSV), exist_ok=True)
        totals.to_csv(TOTALS_CSV, index=False)
    totals['square_id'] = totals['square_id'].astype(int)
    totals['total_cdr'] = pd.to_numeric(totals['total_cdr'], errors='coerce')
    totals = totals.dropna(subset=['total_cdr'])
    return totals


def savefig(path, fig=None):
    if fig is None:
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        return
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def adf_like_test(series, max_lag=1):
    y = np.asarray(series.values, dtype=float)
    if len(y) <= max_lag + 2:
        return {'statistic': np.nan, 'p_value': np.nan, 'n_obs': int(len(y))}

    dy = np.diff(y)
    y_lag = y[:-1]

    rows = len(dy) - max_lag
    X = [np.ones(rows), y_lag[max_lag:]]
    for lag in range(1, max_lag + 1):
        X.append(dy[max_lag - lag: len(dy) - lag])
    X = np.column_stack(X)
    target = dy[max_lag:]

    beta, *_ = np.linalg.lstsq(X, target, rcond=None)
    resid = target - X @ beta
    dof = max(len(target) - X.shape[1], 1)
    sigma2 = float((resid @ resid) / dof)
    xtx_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(sigma2 * xtx_inv))
    stat = float(beta[1] / se[1]) if se[1] > 0 else np.nan

    # Approximate two-sided p-value with the normal distribution tail.
    abs_stat = abs(stat) if np.isfinite(stat) else np.nan
    if np.isnan(abs_stat):
        p_value = np.nan
    else:
        p_value = float(2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs_stat / math.sqrt(2.0)))))

    return {
        'statistic': stat,
        'p_value': p_value,
        'n_obs': int(len(y)),
        'lags': int(max_lag),
    }


def acf_values(series, nlags):
    x = np.asarray(series.values, dtype=float)
    x = x - x.mean()
    denom = np.dot(x, x)
    values = [1.0]
    for lag in range(1, nlags + 1):
        if lag >= len(x):
            values.append(np.nan)
        else:
            values.append(float(np.dot(x[:-lag], x[lag:]) / denom))
    return np.array(values)


def pacf_values(series, nlags):
    acf = acf_values(series, nlags)
    pacf = [1.0]
    for k in range(1, nlags + 1):
        if k == 1:
            pacf.append(acf[1])
            continue
        toeplitz = np.empty((k, k), dtype=float)
        for i in range(k):
            for j in range(k):
                toeplitz[i, j] = acf[abs(i - j)]
        rhs = acf[1 : k + 1]
        try:
            phi = np.linalg.solve(toeplitz, rhs)
            pacf.append(float(phi[-1]))
        except np.linalg.LinAlgError:
            pacf.append(np.nan)
    return np.array(pacf)


def seasonal_decompose_simple(series, period=144):
    trend = series.rolling(period, center=True, min_periods=1).mean()
    detrended = series - trend
    seasonal_pattern = detrended.groupby(np.arange(len(detrended)) % period).transform('mean')
    residual = series - trend - seasonal_pattern
    return pd.DataFrame({'trend': trend, 'seasonal': seasonal_pattern, 'resid': residual})


def plot_pdf(totals):
    fig, ax = plt.subplots(figsize=(9, 5))
    values = totals['total_cdr'].astype(float)
    sns.histplot(values, bins=120, stat='density', kde=True, ax=ax, color='#c04d2d')
    ax.set_title('PDF of total traffic per grid square')
    ax.set_xlabel('Total traffic (two months)')
    ax.set_ylabel('Density')
    savefig(os.path.join(OUT_DIR, 'pdf_total_traffic.png'), fig)


def plot_first_two_weeks(series_map):
    fig, axes = plt.subplots(len(series_map), 1, figsize=(12, 10), sharex=False)
    if len(series_map) == 1:
        axes = [axes]
    for ax, (sid, series) in zip(axes, series_map.items()):
        first_2w = series.iloc[:2016]
        ax.plot(first_2w.index, first_2w.values, lw=1.0, color='#1f77b4')
        ax.set_title(f'Time series over 2 weeks - Square {sid}')
        ax.set_xlabel('Time')
        ax.set_ylabel('CDR')
    savefig(os.path.join(OUT_DIR, 'series_first_two_weeks.png'), fig)


def plot_stationarity(series, square_id):
    window = 144
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(series.index, series.values, color='steelblue', linewidth=0.8, label='CDR')
    ax.plot(rolling_mean.index, rolling_mean.values, color='darkorange', label=f'Rolling mean ({window})')
    ax.plot(rolling_std.index, rolling_std.values, color='forestgreen', label=f'Rolling std ({window})')
    ax.set_title(f'Visual stationarity - Square {square_id}')
    ax.set_xlabel('Time')
    ax.set_ylabel('CDR')
    ax.legend()
    savefig(os.path.join(OUT_DIR, f'stationarity_square_{square_id}.png'), fig)


def plot_decomposition(series, square_id):
    decomposed = seasonal_decompose_simple(series, period=144)
    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(series.index, series.values, color='steelblue', lw=0.8)
    axes[0].set_title(f'Original series - Square {square_id}')
    axes[1].plot(decomposed['trend'].index, decomposed['trend'].values, color='darkorange', lw=0.8)
    axes[1].set_title('Trend')
    axes[2].plot(decomposed['seasonal'].index, decomposed['seasonal'].values, color='forestgreen', lw=0.8)
    axes[2].set_title('Seasonality')
    axes[3].plot(decomposed['resid'].index, decomposed['resid'].values, color='crimson', lw=0.8)
    axes[3].set_title('Residuals')
    fig.suptitle(f'Approximate additive decomposition - Square {square_id}', y=1.02)
    savefig(os.path.join(OUT_DIR, f'decomposition_square_{square_id}.png'))


def plot_acf_pacf(series, square_id):
    lags = 144
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    acf_vals = acf_values(series, lags)
    pacf_vals = pacf_values(series, lags)
    axes[0].stem(range(lags + 1), acf_vals, basefmt=' ', linefmt='C0-', markerfmt='C0o')
    axes[1].stem(range(lags + 1), pacf_vals, basefmt=' ', linefmt='C1-', markerfmt='C1o')
    axes[0].set_title(f'ACF - Square {square_id}')
    axes[1].set_title(f'PACF - Square {square_id}')
    savefig(os.path.join(OUT_DIR, f'acf_pacf_square_{square_id}.png'), fig)


def plot_heatmap(totals):
    grid = np.zeros((100, 100), dtype=float)
    for _, row in totals.iterrows():
        sid = int(row['square_id'])
        value = float(row['total_cdr'])
        idx = sid - 1
        r = idx // 100
        c = idx % 100
        if 0 <= r < 100 and 0 <= c < 100:
            grid[r, c] = value
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(grid, ax=ax, cmap='magma')
    ax.set_title('Heatmap of total traffic on 100x100 grid')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    savefig(os.path.join(OUT_DIR, 'heatmap_grid_total_cdr.png'), fig)


def anomaly_summary(series, square_id):
    rolling = series.rolling(144, center=True)
    med = rolling.median()
    mad = (series - med).abs().rolling(144, center=True).median()
    score = (series - med).abs() / (1.4826 * mad.replace(0, np.nan))
    score = score.replace([np.inf, -np.inf], np.nan)
    top = score.sort_values(ascending=False).head(10)
    out = pd.DataFrame({'timestamp': top.index.astype(str), 'anomaly_score': top.values, 'cdr': series.loc[top.index].values})
    out.to_csv(os.path.join(OUT_DIR, f'anomalies_square_{square_id}.csv'), index=False)
    return out


def main():
    totals = load_totals()
    top_square = int(totals.sort_values('total_cdr', ascending=False).iloc[0]['square_id'])
    square_ids = [top_square, 4159, 4556]

    series_map = {sid: load_square_series(sid) for sid in square_ids}

    plot_pdf(totals)
    plot_first_two_weeks(series_map)

    top_series = series_map[top_square]
    plot_stationarity(top_series, top_square)
    adf = adf_like_test(top_series, max_lag=1)
    plot_decomposition(top_series, top_square)
    plot_acf_pacf(top_series, top_square)
    plot_heatmap(totals)
    anomalies = anomaly_summary(top_series, top_square)

    summary = {
        'top_square': top_square,
        'square_ids': square_ids,
        'adf': adf,
        'top_series_length': int(len(top_series)),
        'top_series_start': str(top_series.index.min()),
        'top_series_end': str(top_series.index.max()),
        'anomaly_top_10': anomalies.to_dict(orient='records'),
        'pdf_stats': {
            'mean_total_cdr': float(totals['total_cdr'].mean()),
            'median_total_cdr': float(totals['total_cdr'].median()),
            'std_total_cdr': float(totals['total_cdr'].std()),
            'min_total_cdr': float(totals['total_cdr'].min()),
            'max_total_cdr': float(totals['total_cdr'].max()),
        },
    }
    with open(os.path.join(OUT_DIR, 'eda_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print('EDA complete')
    print('Top square:', top_square)
    print('ADF:', adf)
    print('Outputs in:', OUT_DIR)


if __name__ == '__main__':
    main()
