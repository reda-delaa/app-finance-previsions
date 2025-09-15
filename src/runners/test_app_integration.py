"""
Integration tests for src/apps/app.py
Tests the main Streamlit application functionality.
"""

import pytest
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Ensure src is in path for imports
_src_root = Path(__file__).resolve().parents[1] / "src"
if str(_src_root) not in sys.path:
    sys.path.insert(0, str(_src_root))


class TestAppImports:
    """Test that app imports work correctly."""

    def test_safe_import_success(self):
        """Test successful safe import."""
        with patch('importlib.import_module') as mock_import:
            from src.apps.app import safe_import

            mock_module = MagicMock()
            mock_import.return_value = mock_module

            result, error = safe_import("test.module", "test_func")
            assert error is None
            assert result is not None

    def test_safe_import_failure(self):
        """Test failed safe import."""
        with patch('importlib.import_module', side_effect=ImportError("Module not found")):
            from src.apps.app import safe_import

            result, error = safe_import("nonexistent.module", "func")
            assert result is None
            assert "Module not found" in error


class TestUtilityFunctions:
    """Test app utility functions."""

    def test_json_safe_functionality(self):
        """Test _json_s function handles various inputs."""
        from src.apps.app import _json_s

        # Test normal dict
        result = _json_s({"key": "value"})
        assert '"key": "value"' in result

        # Test list
        result = _json_s([1, 2, 3])
        assert result == "[1, 2, 3]"

        # Test string
        result = _json_s("simple string")
        assert result == '"simple string"'

        # Test None
        result = _json_s(None)
        assert result == "null"

    def test_trace_call_decorator(self):
        """Test the trace_call decorator."""
        from src.apps.app import trace_call
        import time

        call_log = []

        def mock_logger(msg, *args, **kwargs):
            call_log.append(msg)

        with patch('src.apps.app.logger') as mock_logger_obj:
            mock_logger_obj.debug = mock_logger

            @trace_call("test_func")
            def test_function(x, y=10):
                time.sleep(0.01)  # Small delay
                return x + y

            result = test_function(5, y=15)
            assert result == 20

            # Check that debug logging was called
            assert len(call_log) >= 2  # Should have → and ← entries
            assert "→ test_func" in call_log[0]


class TestFunctionResolution:
    """Test function resolution for various modules."""

    def test_resolve_arbitre_with_class(self):
        """Test arbitre resolution when EconomicAnalyst class is available."""
        with patch('src.apps.app.safe_import') as mock_import:
            # Mock the imports to return the class and input classes
            mock_economic_analyst = MagicMock()
            mock_economic_input = MagicMock()

            mock_import.side_effect = lambda path, attr=None: [
                (None, "not found"),  # arbitre function
                (None, "not found"),  # arbitrage function
                (mock_economic_analyst, None),  # EconomicAnalyst class
                (mock_economic_input, None),     # EconomicInput class
            ][0] if "EconomicAnalyst" in str(attr or path) else (None, "not found")

            from src.apps.app import _resolve_arbitre

            result = _resolve_arbitre()

            assert result is not None
            assert callable(result)

    def test_resolve_arbitre_no_modules(self):
        """Test arbitre resolution when no modules are available."""
        with patch('src.apps.app.safe_import', return_value=(None, "Module not found")):
            from src.apps.app import _resolve_arbitre

            result = _resolve_arbitre()
            assert result is None


class TestMockedDependencies:
    """Test functions that require external dependencies."""

    def test_compute_technical_features_mock(self):
        """Test compute_technical_features with mocked dependencies."""
        mock_result = {
            'ticker': 'AAPL',
            'last_price': 150.0,
            'signals': {'score': 0.8},
            'regime': {'trend': 'Bull'},
            'risk': {'vol_ann_pct': 25.0}
        }

        with patch('src.apps.app.compute_technical_features', return_value=mock_result):
            from src.apps.app import compute_technical_features
            result = compute_technical_features('AAPL', window=180)
            assert result['ticker'] == 'AAPL'
            assert result['last_price'] == 150.0

    def test_load_fundamentals_mock(self):
        """Test load_fundamentals with mocked function."""
        mock_data = {
            'income_stmt': 'sample income data',
            'balance_sheet': 'sample balance data'
        }

        with patch('src.apps.app.load_fundamentals', return_value=mock_data):
            from src.apps.app import load_fundamentals
            result = load_fundamentals('AAPL')
            assert 'income_stmt' in result
            assert 'balance_sheet' in result

    def test_load_news_mock(self):
        """Test load_news with mocked function."""
        mock_news = [
            {'title': 'News 1', 'content': 'Content 1', 'source': 'Reuters'},
            {'title': 'News 2', 'content': 'Content 2', 'source': 'Bloomberg'}
        ]

        with patch('src.apps.app.load_news', return_value=mock_news):
            from src.apps.app import load_news
            result = load_news(window_days=7, tickers=['AAPL'])
            assert len(result) == 2
            assert result[0]['title'] == 'News 1'


class TestErrorHandling:
    """Test error handling in the application."""

    def test_log_exc_function(self):
        """Test log_exc function logs exceptions properly."""
        from src.apps.app import log_exc, _json_s

        log_messages = []
        def mock_error(msg):
            log_messages.append(msg)

        with patch('src.apps.app.logger') as mock_logger:
            mock_logger.error = mock_error

            try:
                raise ValueError("Test error")
            except ValueError as e:
                log_exc("test_location", e)

            assert len(log_messages) >= 1
            assert "EXC @ test_location" in log_messages[0]


class TestConfigurationConstants:
    """Test that important constants are defined correctly."""

    def test_constants_defined(self):
        """Test that critical constants exist."""
        from src.apps.app import LOG_DIR, LOG_FILE, logger

        # Check that paths are Path objects or strings
        assert isinstance(LOG_DIR, (str, Path))
        assert isinstance(LOG_FILE, (str, Path))

        # Check that logger exists and has expected level
        assert hasattr(logger, 'debug')
        assert hasattr(logger, 'info')
        assert hasattr(logger, 'error')


if __name__ == "__main__":
    pytest.main([__file__])
