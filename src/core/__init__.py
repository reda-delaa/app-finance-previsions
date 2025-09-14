"""Core package for financial analysis tools."""

from .config import config
from .models import FinancialMetric, CompanyFinancials, FinancialAnalysis
from .io_utils import (
    setup_logging,
    read_jsonl,
    write_jsonl,
    read_parquet,
    write_parquet,
    Cache,
    get_artifacts_dir
)
