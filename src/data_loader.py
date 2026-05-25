import os
import pandas as pd
import numpy as np


def _rename_chunk_columns(chunk):
    if 'Square id' not in chunk.columns:
        ncols = chunk.shape[1]
        colnames = []
        for c in range(ncols):
            if c == 0:
                colnames.append('Square id')
            elif c == 1:
                colnames.append('Time Interval')
            elif c == 2:
                colnames.append('CDR')
            else:
                colnames.append(f'v{c}')
        chunk.columns = colnames
    return chunk


def _normalize_chunk_types(chunk):
    if 'Square id' in chunk.columns:
        chunk['Square id'] = pd.to_numeric(chunk['Square id'], errors='coerce').astype('int64')
    if 'Time Interval' in chunk.columns:
        chunk['Time Interval'] = pd.to_numeric(chunk['Time Interval'], errors='coerce').astype('int64')
    if 'CDR' in chunk.columns:
        chunk['CDR'] = pd.to_numeric(chunk['CDR'], errors='coerce').astype('float64')
    value_cols = [c for c in chunk.columns if c not in ('Square id', 'Time Interval', 'CDR')]
    for col in value_cols:
        chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
    return chunk


def _prepare_parquet_frame(df):
    df = _normalize_chunk_types(df.copy())
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        if pd.api.types.is_float_dtype(df[col].dtype):
            df[col] = df[col].astype('float64')
        elif pd.api.types.is_integer_dtype(df[col].dtype):
            df[col] = df[col].astype('int64')
    return df

def mem_usage(df):
    return df.memory_usage(deep=True).sum() / 1024**2

def reduce_mem_usage(df, verbose=True):
    start_mem = mem_usage(df)
    for col in df.columns:
        col_type = df[col].dtype
        if col_type == object:
            df[col] = df[col].astype('category')
        elif pd.api.types.is_integer_dtype(col_type):
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min >= 0:
                if c_max < 255:
                    df[col] = df[col].astype(np.uint8)
                elif c_max < 65535:
                    df[col] = df[col].astype(np.uint16)
                elif c_max < np.iinfo(np.uint32).max:
                    df[col] = df[col].astype(np.uint32)
                else:
                    df[col] = df[col].astype(np.uint64)
            else:
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
        elif pd.api.types.is_float_dtype(col_type):
            df[col] = df[col].astype(np.float32)
    end_mem = mem_usage(df)
    if verbose:
        print(f"Memory usage decreased from {start_mem:.2f} MB to {end_mem:.2f} MB")
    return df

def sample_memory_summary(csv_path, nrows=100000):
    """Load a sample, print memory usage before/after, and return the optimized sample."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    print("Loading sample...")
    df = pd.read_csv(csv_path, nrows=nrows)
    before = mem_usage(df)
    print(f"Before optimization: {before:.2f} MB")
    df_opt = reduce_mem_usage(df)
    after = mem_usage(df_opt)
    print(f"After optimization: {after:.2f} MB")
    return df_opt

def csv_to_parquet_partitioned(csv_path, out_dir, chunksize=500000):
    """Read an irregular space-delimited TXT file in chunks, optimize dtypes,
    and write Parquet files partitioned by `Square id`.

    The loader is robust to files without headers: it renames columns to
    `Square id`, `Time Interval`, `CDR`, and `v#` for remaining columns.
    """
    os.makedirs(out_dir, exist_ok=True)
    # Lecture manuelle ligne par ligne pour gérer des lignes avec nombre de champs variable
    chunk_rows = []
    chunk_idx = 0
    with open(csv_path, 'r', encoding='utf-8') as fh:
        for line in fh:
            parts = line.strip().split()
            if not parts:
                continue
            chunk_rows.append(parts)
            if len(chunk_rows) >= chunksize:
                chunk_idx += 1
                chunk = pd.DataFrame(chunk_rows)
                chunk_rows = []

                chunk = _rename_chunk_columns(chunk)
                chunk = _normalize_chunk_types(chunk)
                chunk = reduce_mem_usage(chunk, verbose=False)

                for square, grp in chunk.groupby('Square id'):
                    fname = os.path.join(out_dir, f"square_{int(square)}.parquet")
                    if os.path.exists(fname):
                        existing = pd.read_parquet(fname)
                        combined = pd.concat([_prepare_parquet_frame(existing), _prepare_parquet_frame(grp)], ignore_index=True)
                        combined = _prepare_parquet_frame(combined)
                        combined.to_parquet(fname, index=False)
                    else:
                        _prepare_parquet_frame(grp).to_parquet(fname, index=False)
                print(f"Chunk {chunk_idx} processed and written.")

    # traiter les lignes restantes
    if chunk_rows:
        chunk_idx += 1
        chunk = pd.DataFrame(chunk_rows)
        chunk = _rename_chunk_columns(chunk)
        chunk = _normalize_chunk_types(chunk)
        chunk = reduce_mem_usage(chunk, verbose=False)

        for square, grp in chunk.groupby('Square id'):
            fname = os.path.join(out_dir, f"square_{int(square)}.parquet")
            if os.path.exists(fname):
                existing = pd.read_parquet(fname)
                combined = pd.concat([_prepare_parquet_frame(existing), _prepare_parquet_frame(grp)], ignore_index=True)
                combined = _prepare_parquet_frame(combined)
                combined.to_parquet(fname, index=False)
            else:
                _prepare_parquet_frame(grp).to_parquet(fname, index=False)
        print(f"Chunk {chunk_idx} processed and written.")
