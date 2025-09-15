"""
Tests for src/analytics/phase2_technical.py
Tests technical analysis functions, indicators, signals, and backtesting.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from src.analytics.phase2_technical import (
    load_prices,
    compute_indicators,
    technical_signals,
    detect_regime,
    risk_stats,
    build_technical_view,
    compute_technical_features
)


class TestDataLoading:
    """Test data loading functions."""

    @patch('src.analytics.phase2_technical.yf')
    def test_load_prices_success(self, mock_yf):
        """Test successful price loading."""
        # Mock yfinance response
        mock_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [95, 96, 97],
            'Close': [103, 104, 105],
            'Volume': [1000, 1100, 1200]
        }, index=pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']))

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = mock_data
        mock_yf.Ticker.return_value = mock_ticker

        result = load_prices('AAPL', period='1mo', interval='1d')

        assert not result.empty
        assert len(result) == 3
        assert list(result.columns) == ['Open', 'High', 'Low', 'Close', 'Volume']
        assert result.index.tz is None  # Should be timezone-naive

    @patch('src.analytics.phase2_technical.yf')
    def test_load_prices_empty(self, mock_yf):
        """Test empty price data handling."""
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()
        mock_yf.Ticker.return_value = mock_ticker

        result = load_prices('INVALID')
        assert result.empty


class TestIndicators:
    """Test technical indicators computation."""

    def create_sample_data(self):
        """Create sample OHLCV data for testing."""
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        np.random.seed(42)  # For reproducibility

        close = 100 + np.cumsum(np.random.randn(100) * 2)
        high = close + np.abs(np.random.randn(100) * 3)
        low = close - np.abs(np.random.randn(100) * 3)
        open_prices = close + np.random.randn(100) * 1
        volume = np.random.randint(1000, 10000, 100)

        df = pd.DataFrame({
            'Open': open_prices,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': volume
        }, index=dates)

        return df

    def test_compute_indicators(self):
        """Test indicators computation."""
        sample_data = self.create_sample_data()
        result = compute_indicators(sample_data)

        assert isinstance(result, object)  # IndicatorSet
        assert not result.df.empty
        assert 'SMA_20' in result.df.columns
        assert 'RSI_14' in result.df.columns
        assert 'MACD' in result.df.columns
        assert 'BB_Upper' in result.df.columns
        assert 'ADX_14' in result.df.columns

        # Check metadata
        assert 'indicator_set' in result.meta
        assert 'rows' in result.meta

    def test_indicator_set_to_dict(self):
        """Test IndicatorSet to_dict method."""
        sample_data = self.create_sample_data()
        indicator_set = compute_indicators(sample_data)

        result = indicator_set.to_dict()
        assert 'columns' in result
        assert 'rows' in result
        assert isinstance(result['columns'], list)


class TestSignals:
    """Test technical signals calculation."""

    def test_technical_signals(self):
        """Test technical signals with complete data."""
        # Create minimal data with all required columns
        dates = pd.date_range('2024-01-01', periods=50, freq='D')

        df = pd.DataFrame({
            'Close': [100] * 50,
            'High': [105] * 50,
            'Low': [95] * 50,
            'Open': [100] * 50,
            'Volume': [1000] * 50,
            'RSI_14': [65] * 50,
            'MACD': [1.5] * 50,
            'MACD_Signal': [1.2] * 50,
            'EMA_12': [102] * 50,
            'EMA_26': [100] * 50,
            'SMA_20': [101] * 50,
            'SMA_50': [99] * 50,
            'SMA_200': [98] * 50,
            'Trend200_slope': [0.5] * 50,
            'BB_Upper': [110] * 50,
            'BB_Lower': [90] * 50,
            'BB_Middle': [100] * 50,
            'Donchian20_Up': [107] * 50,
            'Donchian20_Down': [93] * 50,
            'ADX_14': [28] * 50
        }, index=dates)

        from src.analytics.phase2_technical import IndicatorSet
        indicators = IndicatorSet(df=df, meta={})

        result = technical_signals(indicators)

        assert isinstance(result, object)  # TechnicalSignals
        assert isinstance(result.score, float)
        assert result.score >= -1.0 and result.score <= 1.0
        assert isinstance(result.components, dict)
        assert isinstance(result.labels, list)

    def test_signals_to_dict(self):
        """Test TechnicalSignals to_dict method."""
        # Create minimal data
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        df = pd.DataFrame({
            'Close': [100] * 30, 'High': [105] * 30, 'Low': [95] * 30,
            'Open': [100] * 30, 'Volume': [1000] * 30,
            'RSI_14': [50] * 30, 'MACD': [0] * 30, 'MACD_Signal': [0] * 30,
            'EMA_12': [100] * 30, 'EMA_26': [100] * 30,
            'SMA_20': [100] * 30, 'SMA_50': [100] * 30, 'SMA_200': [100] * 30,
            'Trend200_slope': [0] * 30, 'BB_Upper': [105] * 30,
            'BB_Lower': [95] * 30, 'BB_Middle': [100] * 30,
            'Donchian20_Up': [105] * 30, 'Donchian20_Down': [95] * 30,
            'ADX_14': [15] * 30
        }, index=dates)

        from src.analytics.phase2_technical import IndicatorSet
        indicators = IndicatorSet(df=df, meta={})
        signals = technical_signals(indicators)

        result = signals.to_dict()
        assert 'score' in result
        assert 'components' in result
        assert 'labels' in result


class TestRegimeDetection:
    """Test regime detection functions."""

    def test_detect_regime_bull(self):
        """Test bull regime detection."""
        dates = pd.date_range('2024-01-01', periods=250, freq='D')

        # Create data indicating bull regime
        close = [100 + i*0.1 for i in range(250)]  # Steady upward trend
        df = pd.DataFrame({
            'Close': close,
            'SMA_200': [95 + i*0.05 for i in range(250)],
            'Trend200_slope': [0.05] * 250
        }, index=dates)

        from src.analytics.phase2_technical import IndicatorSet
        indicators = IndicatorSet(df=df, meta={})

        result = detect_regime(indicators)

        assert result.trend == "Bull"
        assert result.vol_regime in ["LowVol", "HighVol"]
        assert result.slope200 is not None

    def test_detect_regime_bear(self):
        """Test bear regime detection."""
        dates = pd.date_range('2024-01-01', periods=250, freq='D')

        # Create data indicating bear regime
        close = [100 - i*0.1 for i in range(250)]  # Steady downward trend
        df = pd.DataFrame({
            'Close': close,
            'SMA_200': [105 - i*0.05 for i in range(250)],
            'Trend200_slope': [-0.05] * 250
        }, index=dates)

        from src.analytics.phase2_technical import IndicatorSet
        indicators = IndicatorSet(df=df, meta={})

        result = detect_regime(indicators)

        assert result.trend == "Bear"
        assert result.vol_regime in ["LowVol", "HighVol"]

    def test_detect_regime_range(self):
        """Test range regime detection."""
        dates = pd.date_range('2024-01-01', periods=250, freq='D')

        # Create data indicating range regime
        close = [100 + 2*np.sin(2*np.pi*i/50) for i in range(250)]  # Oscillating around 100
        df = pd.DataFrame({
            'Close': close,
            'SMA_200': [100] * 250,
            'Trend200_slope': [0.001] * 250
        }, index=dates)

        from src.analytics.phase2_technical import IndicatorSet
        indicators = IndicatorSet(df=df, meta={})

        result = detect_regime(indicators)

        assert result.trend == "Range"
        assert result.vol_regime in ["LowVol", "HighVol"]

    def test_detect_regime_insufficient_data(self):
        """Test regime detection with insufficient data."""
        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        df = pd.DataFrame({'Close': [100] * 50}, index=dates)

        from src.analytics.phase2_technical import IndicatorSet
        indicators = IndicatorSet(df=df, meta={})

        result = detect_regime(indicators)

        assert result.trend == "Range"
        assert result.vol_regime == "LowVol"


class TestRiskStats:
    """Test risk statistics calculation."""

    def test_risk_stats_sufficient_data(self):
        """Test risk stats with sufficient data."""
        dates = pd.date_range('2024-01-01', periods=50, freq='D')

        # Create price data with some volatility
        close = 100 + np.random.randn(50).cumsum() * 2

        df = pd.DataFrame({'Close': close}, index=dates)

        result = risk_stats(df)

        assert result.vol_ann_pct is not None
        assert result.var95_pct is not None
        assert result.max_dd_pct is not None

        assert result.vol_ann_pct >= 0  # Volatility should be positive
        assert result.max_dd_pct <= 0  # Max drawdown should be negative or zero

    def test_risk_stats_insufficient_data(self):
        """Test risk stats with insufficient data."""
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        df = pd.DataFrame({'Close': [100] * 10}, index=dates)

        result = risk_stats(df)

        assert result.vol_ann_pct is None
        assert result.var95_pct is None
        assert result.max_dd_pct is None

    def test_risk_stats_to_dict(self):
        """Test RiskStats to_dict method."""
        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        close = 100 + np.random.randn(50).cumsum() * 2
        df = pd.DataFrame({'Close': close}, index=dates)

        stats = risk_stats(df)
        result = stats.to_dict()

        assert isinstance(result, dict)
        assert len(result) == 3


class TestHighLevelAPI:
    """Test high-level API functions."""

    @patch('src.analytics.phase2_technical.build_technical_view')
    def test_compute_technical_features_success(self, mock_build):
        """Test successful technical features computation."""
        mock_result = {
            'ticker': 'AAPL',
            'last_price': 150.0,
            'signals': {'score': 0.8},
            'regime': {'trend': 'Bull'},
            'risk': {'vol_ann_pct': 25.0}
        }
        mock_build.return_value = mock_result

        result = compute_technical_features('AAPL', window=180)

        assert result['ticker'] == 'AAPL'
        assert result['last_price'] == 150.0
        assert result['signals']['score'] == 0.8
        assert result['window'] == 180

    @patch('src.analytics.phase2_technical.build_technical_view')
    def test_compute_technical_features_error(self, mock_build):
        """Test error handling in technical features computation."""
        mock_build.side_effect = Exception("API Error")

        result = compute_technical_features('AAPL', window=180)

        assert result['ticker'] == 'AAPL'
        assert 'error' in result
        assert 'Failed to compute' in result['error']

    @patch('src.analytics.phase2_technical.load_prices')
    @patch('src.analytics.phase2_technical.compute_indicators')
    @patch('src.analytics.phase2_technical.technical_signals')
    @patch('src.analytics.phase2_technical.detect_regime')
    @patch('src.analytics.phase2_technical.risk_stats')
    def test_build_technical_view_success(self, mock_risk, mock_regime, mock_signals, mock_indicators, mock_load):
        """Test successful technical view building."""
        # Mock the dependencies
        mock_px = pd.DataFrame({
            'Close': [100, 101, 102]
        }, index=pd.date_range('2024-01-01', periods=3))

        mock_load.return_value = mock_px
        mock_indicators.return_value = MagicMock()

        mock_signals_obj = MagicMock()
        mock_signals_obj.to_dict.return_value = {'score': 0.7}
        mock_signals.return_value = mock_signals_obj

        mock_regime_obj = MagicMock()
        mock_regime_obj.to_dict.return_value = {'trend': 'Bull'}
        mock_regime.return_value = mock_regime_obj

        mock_risk_obj = MagicMock()
        mock_risk_obj.to_dict.return_value = {'vol_ann_pct': 15.0}
        mock_risk.return_value = mock_risk_obj

        result = build_technical_view('AAPL')

        assert result['ticker'] == 'AAPL'
        assert isinstance(result['last_price'], float)
        assert result['signals'] == {'score': 0.7}
        assert result['regime'] == {'trend': 'Bull'}
        assert result['risk'] == {'vol_ann_pct': 15.0}

    @patch('src.analytics.phase2_technical.load_prices')
    def test_build_technical_view_no_data(self, mock_load):
        """Test technical view building with no data."""
        mock_load.return_value = pd.DataFrame()

        result = build_technical_view('INVALID')

        assert result['ticker'] == 'INVALID'
        assert 'error' in result
        assert 'No price data' in result['error']


if __name__ == "__main__":
    pytest.main([__file__])
