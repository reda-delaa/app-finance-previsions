"""
Core I/O utilities for financial data processing.
Handles JSONL, parquet, caching, and time-related operations.
"""

import json
import os
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

# Configure logging
def setup_logging(name: str = "finance_analysis") -> logging.Logger:
    """Configure rotating file logger"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # File handler
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        fh = logging.FileHandler(log_dir / f"{name}.log")
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
    
    return logger

logger = setup_logging()

def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists and return Path object"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def read_jsonl(path: Union[str, Path]) -> List[Dict]:
    """Read JSONL file and return list of dictionaries"""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def write_jsonl(data: List[Dict], path: Union[str, Path]) -> None:
    """Write list of dictionaries to JSONL file"""
    ensure_dir(Path(path).parent)
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def read_parquet(path: Union[str, Path]) -> pd.DataFrame:
    """Read Parquet file with basic error handling"""
    try:
        return pd.read_parquet(path)
    except Exception as e:
        logger.error(f"Error reading parquet file {path}: {e}")
        return pd.DataFrame()

def write_parquet(df: pd.DataFrame, path: Union[str, Path]) -> None:
    """Write DataFrame to Parquet with basic error handling"""
    try:
        ensure_dir(Path(path).parent)
        df.to_parquet(path, index=True)
    except Exception as e:
        logger.error(f"Error writing parquet file {path}: {e}")

class Cache:
    """Simple file-based cache with TTL support"""
    
    def __init__(self, cache_dir: Union[str, Path] = "cache"):
        self.cache_dir = ensure_dir(cache_dir)
    
    def _get_cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"
    
    def get(self, key: str, ttl_hours: Optional[int] = None) -> Optional[Any]:
        """Get value from cache, None if missing or expired"""
        path = self._get_cache_path(key)
        if not path.exists():
            return None
            
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            if ttl_hours is not None:
                cached_time = datetime.fromisoformat(data['_timestamp'])
                age_hours = (datetime.now(timezone.utc) - cached_time).total_seconds() / 3600
                if age_hours > ttl_hours:
                    return None
            return data['value']
        except Exception as e:
            logger.error(f"Cache read error for {key}: {e}")
            return None
    
    def set(self, key: str, value: Any) -> None:
        """Set cache value with current timestamp"""
        try:
            data = {
                '_timestamp': datetime.now(timezone.utc).isoformat(),
                'value': value
            }
            path = self._get_cache_path(key)
            path.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
        except Exception as e:
            logger.error(f"Cache write error for {key}: {e}")

def get_artifacts_dir(name: str) -> Path:
    """Get dated artifacts directory for outputs"""
    base = ensure_dir("artifacts")
    date = datetime.now().strftime("%Y-%m-%d")
    return ensure_dir(base / f"{date}_{name}")

# Time utilities
def ensure_utc(dt: datetime) -> datetime:
    """Ensure datetime is UTC-aware"""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def parse_iso_date(date_str: str) -> datetime:
    """Parse ISO date string to UTC datetime"""
    try:
        dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        return ensure_utc(dt)
    except Exception as e:
        logger.error(f"Error parsing date {date_str}: {e}")
        return datetime.now(timezone.utc)
