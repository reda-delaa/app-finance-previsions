"""
Configuration management for financial analysis tools.
Loads settings from environment variables and config files.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

from .io_utils import logger

# Load environment variables from .env file
load_dotenv()

class Config:
    """Global configuration singleton"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize configuration from environment"""
        # Base paths
        self.base_dir = Path(os.getenv('FINANCE_BASE_DIR', Path.cwd()))
        self.data_dir = self.base_dir / 'data'
        self.cache_dir = self.base_dir / 'cache'
        self.artifacts_dir = self.base_dir / 'artifacts'
        
        # Ensure directories exist
        for dir_path in [self.data_dir, self.cache_dir, self.artifacts_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # API Keys and credentials (prefer secrets_local.get_key when available)
        try:
            try:
                from secrets_local import get_key as _get_key  # type: ignore
            except Exception:
                from src.secrets_local import get_key as _get_key  # type: ignore
            # support both FINNHUB_API_KEY and FINNHUB_KEY
            self.finnhub_key = (_get_key('FINNHUB_KEY') or _get_key('FINNHUB_API_KEY') or
                                os.getenv('FINNHUB_KEY') or os.getenv('FINNHUB_API_KEY'))
            # Alpha Vantage API key
            self.alpha_vantage_key = (_get_key('ALPHA_VANTAGE_KEY') or _get_key('ALPHA_VANTAGE_API_KEY') or
                                      os.getenv('ALPHA_VANTAGE_KEY') or os.getenv('ALPHA_VANTAGE_API_KEY'))
            # Yahoo Finance API key (optional, mainly for premium features)
            self.yahoo_api_key = (_get_key('YAHOO_API_KEY') or os.getenv('YAHOO_API_KEY'))
        except Exception:
            self.finnhub_key = (os.getenv('FINNHUB_KEY') or os.getenv('FINNHUB_API_KEY'))
            self.alpha_vantage_key = (os.getenv('ALPHA_VANTAGE_KEY') or os.getenv('ALPHA_VANTAGE_API_KEY'))
            self.yahoo_api_key = os.getenv('YAHOO_API_KEY')
        
        # Analysis parameters
        self.default_lookback_years = int(os.getenv('DEFAULT_LOOKBACK_YEARS', '3'))
        self.min_peer_market_cap = float(os.getenv('MIN_PEER_MARKET_CAP', '100')) # millions
        self.max_peer_count = int(os.getenv('MAX_PEER_COUNT', '10'))
        
        # Cache settings
        self.cache_ttl_hours = int(os.getenv('CACHE_TTL_HOURS', '24'))
        
        # Validate required settings
        self._validate_config()
    
    def _validate_config(self):
        """Validate required configuration settings"""
        missing_keys = []

        # Check required API keys
        if not any([self.alpha_vantage_key, self.yahoo_api_key, self.finnhub_key]):
            missing_keys.append("At least one financial API key is required")

        if missing_keys:
            # Log as warning and mark config as invalid; avoid raising to keep import-time safe
            logger.warning(f"Missing required configuration: {', '.join(missing_keys)}")
            # attach a flag so callers can detect missing keys at runtime
            self._is_valid = False
        else:
            self._is_valid = True
    
    def get_data_path(self, *parts: str) -> Path:
        """Get path under data directory"""
        return self.data_dir.joinpath(*parts)
    
    def get_cache_path(self, *parts: str) -> Path:
        """Get path under cache directory"""
        return self.cache_dir.joinpath(*parts)
    
    def get_artifacts_path(self, *parts: str) -> Path:
        """Get path under artifacts directory"""
        return self.artifacts_dir.joinpath(*parts)
    
    @property
    def has_premium_apis(self) -> bool:
        """Check if premium API keys are configured"""
        return bool(self.alpha_vantage_key or self.yahoo_api_key)
    
    def to_dict(self, exclude_secrets: bool = True) -> Dict[str, Any]:
        """Convert config to dictionary, optionally excluding sensitive values"""
        config_dict = {
            'base_dir': str(self.base_dir),
            'data_dir': str(self.data_dir),
            'cache_dir': str(self.cache_dir),
            'artifacts_dir': str(self.artifacts_dir),
            'default_lookback_years': self.default_lookback_years,
            'min_peer_market_cap': self.min_peer_market_cap,
            'max_peer_count': self.max_peer_count,
            'cache_ttl_hours': self.cache_ttl_hours,
            'has_premium_apis': self.has_premium_apis
        }
        
        if not exclude_secrets:
            config_dict.update({
                'finnhub_key': self.finnhub_key
            })
        
        return config_dict


# Global config instance
# Charger des secrets locaux éventuels AVANT la validation (peu importe PYTHONPATH)
try:
    import secrets_local as _secrets  # noqa: F401
except Exception:
    try:
        import src.secrets_local as _secrets  # noqa: F401
    except Exception:
        _secrets = None  # ok si absent

try:
    config = Config()
except Exception as e:
    # Log missing configuration but avoid hard failure at import time so modules
    # remain importable in dev/test environments. If secrets_local provided
    # env vars, Config() will validate properly.
    logger.error("Missing required configuration: %s", e)
    logger.warning(f"Config not initialized at import time: {e}")
    config = None
