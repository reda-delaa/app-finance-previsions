from __future__ import annotations

"""
Lightweight data store utilities for Parquet (analytics lake) and DuckDB queries.

Usage:
- write_parquet(df, path): create directories and write a Parquet file
- have_files(glob): True if any files match the glob pattern
- query_duckdb(sql): run an ad-hoc SQL query (DuckDB) and return a DataFrame
"""

from pathlib import Path
from typing import Optional
import pandas as pd
import duckdb


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_parquet(df: pd.DataFrame, path: str | Path, **kwargs) -> Path:
    p = Path(path)
    ensure_parent(p)
    df.to_parquet(str(p), engine=kwargs.get("engine", "pyarrow"), compression=kwargs.get("compression", "snappy"))
    return p


def have_files(glob_pattern: str) -> bool:
    return any(Path().glob(glob_pattern))


def query_duckdb(sql: str) -> pd.DataFrame:
    con = duckdb.connect(database=":memory:")
    try:
        return con.execute(sql).fetch_df()
    finally:
        con.close()

