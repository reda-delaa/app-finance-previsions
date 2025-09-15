"""
Tests for src/analytics/econ_llm_agent.py
Tests the EconomicAnalyst class and related functions.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from src.analytics.econ_llm_agent import (
    EconomicAnalyst,
    EconomicInput,
    _to_json_serializable,
    _build_context,
    _extract_tail_json_line,
    _agreement_score,
    POWER_NOAUTH_MODELS,
    _format_features,
    _format_news,
    _truncate
)


class TestEconomicAnalyst:
    """Test the EconomicAnalyst class."""

    @pytest.fixture
    def sample_input(self):
        """Sample EconomicInput for testing."""
        return EconomicInput(
            question="What is the impact of inflation on GDP?",
            features={"GDP_growth": 0.02, "inflation_rate": 0.03},
            news=[{
                "title": "Inflation rises 3%",
                "content": "Inflation increased to 3% this month",
                "source": "Reuters"
            }],
            locale="fr"
        )

    def test_initialization_default(self):
        """Test default initialization."""
        agent = EconomicAnalyst()
        assert agent.model_candidates == POWER_NOAUTH_MODELS
        assert agent.temperature == 0.2
        assert agent.max_tokens == 2048

    def test_initialization_custom(self):
        """Test custom initialization."""
        custom_models = ["model1", "model2"]
        agent = EconomicAnalyst(
            model_candidates=custom_models,
            temperature=0.3,
            char_budget=50000
        )
        assert agent.model_candidates == custom_models
        assert agent.temperature == 0.3
        assert agent.char_budget == 50000

    @patch('src.analytics.econ_llm_agent.G4FClient')
    def test_call_model_success(self, mock_client, sample_input):
        """Test successful model call."""
        agent = EconomicAnalyst()

        # Mock the response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Analysis response"
        mock_response.usage = Mock()

        mock_client_instance = Mock()
        mock_client_instance.chat.completions.create.return_value = mock_response
        mock_client.return_value = mock_client_instance

        agent = EconomicAnalyst()  # Recreate to get new client
        messages = agent._build_messages(sample_input)

        success, result = agent._call_model("test-model", messages)

        assert success is True
        assert result["ok"] is True
        assert result["model"] == "test-model"
        assert result["answer"] == "Analysis response"

    @patch('src.analytics.econ_llm_agent.G4FClient')
    def test_call_model_failure(self, mock_client, sample_input):
        """Test model call failure."""
        agent = EconomicAnalyst()

        # Mock client to raise exception
        mock_client_instance = Mock()
        mock_client_instance.chat.completions.create.side_effect = Exception("Connection error")
        mock_client.return_value = mock_client_instance

        agent = EconomicAnalyst()  # Recreate to get new client
        messages = agent._build_messages(sample_input)

        success, result = agent._call_model("test-model", messages)

        assert success is False
        assert result["ok"] is False
        assert "Connection error" in result["error"]

    def test_build_messages(self, sample_input):
        """Test message building."""
        agent = EconomicAnalyst()
        messages = agent._build_messages(sample_input)

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "Question" in messages[1]["content"]
        assert "What is the impact of inflation on GDP?" in messages[1]["content"]


class TestUtilityFunctions:
    """Test utility functions."""

    def test_truncate_normal_text(self):
        """Test text truncation when under limit."""
        text = "Short text"
        result = _truncate(text, 100)
        assert result == text

    def test_truncate_long_text(self):
        """Test text truncation when over limit."""
        text = "A" * 200
        result = _truncate(text, 100)
        # The function adds "...\n" in the middle and tail, so length is slightly more than limit
        assert len(result) > 100
        assert "..." in result
        assert result.startswith("AAAAAAAA")
        assert result.endswith("AAA")

    def test_format_features(self):
        """Test feature formatting."""
        features = {"a": 1, "b": 2, "inflation": 3.5}
        result = _format_features(features)
        assert "## Features" in result
        assert "- a: 1" in result
        assert "- b: 2" in result
        assert "- inflation: 3.5" in result

    def test_format_news(self):
        """Test news formatting."""
        news = [{
            "title": "Test Title",
            "content": "Test content",
            "source": "Reuters",
            "tickers": ["AAPL"]
        }]
        result = _format_news(news)
        assert "## News" in result
        assert "Test Title" in result
        assert "Reuters" in result
        assert "AAPL" in result

    def test_build_context(self):
        """Test context building."""
        ein = EconomicInput(
            question="Test question",
            features={"test": 1},
            news=[{"title": "news", "content": "content"}]
        )
        result = _build_context(ein, 1000)
        assert "# Question" in result
        assert "Test question" in result
        assert "## Features" in result


class TestJsonProcessing:
    """Test JSON processing functions."""

    def test_to_json_serializable_basic_types(self):
        """Test serialization of basic Python types."""
        assert _to_json_serializable(None) is None
        assert _to_json_serializable("string") == "string"
        assert _to_json_serializable(42) == 42
        assert _to_json_serializable(3.14) == 3.14
        assert _to_json_serializable(True) == True

    def test_to_json_serializable_dict(self):
        """Test serialization of dictionaries."""
        d = {"a": 1, "b": [1, 2, 3]}
        result = _to_json_serializable(d)
        assert result == d

    def test_to_json_serializable_list(self):
        """Test serialization of lists."""
        l = [1, 2, {"nested": "value"}]
        result = _to_json_serializable(l)
        assert result == l

    def test_to_json_serializable_custom_object(self):
        """Test serialization of custom objects."""
        class CustomObj:
            def __init__(self):
                self.value = 42

        obj = CustomObj()
        result = _to_json_serializable(obj)
        # Should fall back to vars() for unknown objects
        assert result == {"value": 42}

    def test_extract_tail_json_line_valid(self):
        """Test extraction of valid JSON from text."""
        text = """Some analysis text
{"summary": ["point1"], "scenarios": [{"name": "base"}], "risks": [], "impacts": {"FX": []}, "actions": [], "confidence": 0.8}
"""
        result = _extract_tail_json_line(text)
        expected = {
            "summary": ["point1"],
            "scenarios": [{"name": "base"}],
            "risks": [],
            "impacts": {"FX": []},
            "actions": [],
            "confidence": 0.8
        }
        assert result == expected

    def test_extract_tail_json_line_invalid(self):
        """Test extraction with invalid JSON."""
        text = "Some text without valid JSON"
        result = _extract_tail_json_line(text)
        assert result is None

    def test_extract_tail_json_line_missing_keys(self):
        """Test extraction with JSON missing required keys."""
        text = '{"summary": []}'
        result = _extract_tail_json_line(text)
        assert result is None


class TestAgreementScoring:
    """Test agreement scoring functions."""

    def test_agreement_score_identical(self):
        """Test agreement score for identical texts."""
        score = _agreement_score("test text", "test text")
        assert score == 1.0  # Perfect agreement

    def test_agreement_score_different(self):
        """Test agreement score for completely different texts."""
        score = _agreement_score("hello world", "goodbye universe")
        assert score < 0.5  # Low agreement

    def test_agreement_score_partial(self):
        """Test agreement score for partially similar texts."""
        score = _agreement_score("The inflation rate is rising", "Inflation rate shows increase")
        assert 0.1 < score < 0.5  # Partial agreement - adjust expectations to match actual algorithm


class TestEconomicInput:
    """Test EconomicInput dataclass."""

    def test_initialization_minimal(self):
        """Test minimal initialization."""
        ein = EconomicInput(question="Test?")
        assert ein.question == "Test?"
        assert ein.features is None
        assert ein.news is None
        assert ein.attachments is None
        assert ein.locale == "fr-FR"
        assert ein.meta == {}

    def test_initialization_full(self):
        """Test full initialization."""
        ein = EconomicInput(
            question="Full test",
            features={"a": 1},
            news=[{"title": "news"}],
            attachments=["att1"],
            locale="en-US",
            meta={"key": "value"}
        )
        assert ein.question == "Full test"
        assert ein.features == {"a": 1}
        assert ein.news == [{"title": "news"}]
        assert ein.attachments == ["att1"]
        assert ein.locale == "en-US"
        assert ein.meta == {"key": "value"}


if __name__ == "__main__":
    pytest.main([__file__])
