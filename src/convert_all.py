import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import csv_to_parquet_partitioned


def main():
    in_dir = os.path.join('data', 'raw', 'EGZHFV')
    out_dir = os.path.join('data', 'processed', 'parquet')
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.exists(in_dir):
        print(f"Input directory not found: {in_dir}. Skipping conversion.")
        return

    files = sorted([os.path.join(in_dir, f) for f in os.listdir(in_dir) if f.endswith('.txt')])
    if not files:
        print(f"No .txt files found in {in_dir}. Nothing to convert.")
        return

    for f in files:
        print(f"Converting {f}...")
        try:
            csv_to_parquet_partitioned(f, out_dir, chunksize=500000)
        except Exception as exc:
            print(f"Failed to convert {f}: {exc}")
            # continue converting other files instead of aborting
            continue


if __name__ == '__main__':
    main()
