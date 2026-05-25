"""
Demo script showcasing the full Milan traffic forecasting pipeline.
Usage: python scripts/demo.py
"""
import os
import sys
import json
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

sns.set_theme(style='whitegrid')
plt.rcParams['figure.dpi'] = 100


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def demo_data_overview():
    """Show data overview."""
    print_section("DATA OVERVIEW")
    
    totals_path = PROJECT_ROOT / 'artifacts' / 'per_square_totals.csv'
    if totals_path.exists():
        totals = pd.read_csv(totals_path)
        print(f"\nTotal squares in dataset: {len(totals)}")
        print(f"\nTop 5 busiest squares by CDR volume:")
        top5 = totals.nlargest(5, 'total_cdr')
        for idx, (_, row) in enumerate(top5.iterrows(), 1):
            print(f"  {idx}. Square {int(row['square_id'])}: {row['total_cdr']:.0f} CDR calls")


def demo_eda_results():
    """Show EDA results."""
    print_section("EXPLORATORY DATA ANALYSIS")
    
    eda_path = PROJECT_ROOT / 'artifacts' / 'eda' / 'eda_summary.json'
    if eda_path.exists():
        with open(eda_path) as f:
            eda = json.load(f)
        
        print(f"\nTop square analyzed: {eda['top_square']}")
        print(f"Time series length: {eda['top_series_length']} intervals (~{eda['top_series_length']/144:.1f} days)")
        print(f"Date range: {eda['top_series_start']} to {eda['top_series_end']}")
        
        print(f"\nADF Test Results (stationarity):")
        print(f"  Statistic: {eda['adf']['statistic']:.4f}")
        print(f"  P-value: {eda['adf']['p_value']:.6f}")
        print(f"  Interpretation: {'STATIONARY ✓' if eda['adf']['p_value'] < 0.05 else 'NON-STATIONARY'}")
        
        print(f"\nTop 3 Anomalies Detected:")
        for i, anomaly in enumerate(eda['anomaly_top_10'][:3], 1):
            print(f"  {i}. {anomaly['timestamp']}: CDR={anomaly['cdr']:.0f} (score: {anomaly['anomaly_score']:.2f})")
        
        print(f"\nTraffic Statistics for Square {eda['top_square']}:")
        stats = eda['pdf_stats']
        print(f"  Mean: {stats['mean_total_cdr']:.0f}")
        print(f"  Median: {stats['median_total_cdr']:.0f}")
        print(f"  Std Dev: {stats['std_total_cdr']:.0f}")
        print(f"  Min: {stats['min_total_cdr']:.0f}")
        print(f"  Max: {stats['max_total_cdr']:.0f}")
def demo_model_results():
    """Show basic model performance summary if available."""
    print_section("MODEL PERFORMANCE SUMMARY")
    perf_path = PROJECT_ROOT / 'artifacts' / 'models' / 'models_performance.csv'
    if perf_path.exists():
        df = pd.read_csv(perf_path)
        print("Top results by MAE:")
        try:
            top = df.sort_values('MAE').head(6)
            print(top[['square', 'model', 'MAE', 'RMSE', 'MAPE']].to_string(index=False))
        except Exception:
            print(df.head().to_string(index=False))
    else:
        print("No model performance CSV found. Run scripts/train_models.py to generate results.")


def demo_prediction_samples():
    """Show sample predictions."""
    print_section("SAMPLE PREDICTIONS")
    
    pred_path = PROJECT_ROOT / 'artifacts' / 'models' / 'square_9510_test_predictions.csv'
    if pred_path.exists():
        preds = pd.read_csv(pred_path, nrows=10)
        
        print("\nFirst 10 predictions for Square 9510 (test set):")
        # show baseline, ridge, and mlp predictions
        cols = ['timestamp', 'actual']
        if 'pred_persistence' in preds.columns:
            cols.append('pred_persistence')
        if 'pred_ridge_seasonal' in preds.columns:
            cols.append('pred_ridge_seasonal')
        if 'pred_mlp_seasonal' in preds.columns:
            cols.append('pred_mlp_seasonal')
        print(preds[cols].to_string(index=False))


def demo_file_structure():
    """Show project structure."""
    print_section("PROJECT STRUCTURE")
    
    print(f"\nProject root: {PROJECT_ROOT}")
    
    key_files = [
        ("README.md", "Main documentation"),
        ("requirements.txt", "Python dependencies"),
        ("scripts/", "Execution scripts"),
        ("scripts/download_instructions.md", "Data download guide"),
        ("src/", "Core modules (loader, conversion)"),
        ("data/raw/", "Raw source data (TXT + geojson)"),
        ("data/processed/parquet/", "Processed Parquet files"),
        ("artifacts/eda/", "EDA outputs (plots, summary)"),
        ("artifacts/models/", "Model predictions & metrics"),
    ]
    
    print("\nKey files and directories:")
    for path, desc in key_files:
        full_path = PROJECT_ROOT / path
        exists = "✓" if full_path.exists() else "✗"
        print(f"  {exists} {path:40} - {desc}")


def demo_next_steps():
    """Print next steps."""
    print_section("NEXT STEPS & AVAILABLE SCRIPTS")
    print("""
The project is fully functional. Here are the available scripts:

1. DATA CONVERSION:
    python src/convert_all.py
    → Converts raw TXT files to Parquet format

2. EXPLORATORY ANALYSIS:
    python scripts/run_eda.py
    → Generates EDA outputs: plots and summaries

3. MODEL TRAINING & EVALUATION:
    python scripts/train_models.py
    → Trains models and evaluates on test set

4. DIAGNOSTIC PLOTS:
    python scripts/plot_model_diagnostics.py
    → Creates residual plots and prediction visualizations

RUNNING THE FULL PIPELINE:
    1. python src/convert_all.py
    2. python scripts/run_eda.py
    3. python scripts/train_models.py
    4. python scripts/plot_model_diagnostics.py
    """)


def main():
    """Run the full demo."""
    print("\n" + "=" * 70)
    print("  MILAN MOBILE TRAFFIC FORECASTING - PIPELINE DEMO")
    print("=" * 70)
    
    demo_data_overview()
    demo_eda_results()
    demo_model_results()
    demo_prediction_samples()
    demo_file_structure()
    demo_next_steps()
    
    print("\n" + "=" * 70)
    print("  Demo complete!")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
