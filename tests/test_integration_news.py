import os
import sys
import pathlib
import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"


@pytest.mark.integration
def test_finnews_run_pipeline_integration():
    # Ces tests nécessitent le réseau et des clés. Opt-in via env var.
    if not os.getenv("AF_ALLOW_INTERNET"):
        pytest.skip("Set AF_ALLOW_INTERNET=1 to run network integration tests")

    sys.path.insert(0, str(PROJECT_SRC))
    mod = __import__("ingestion.finnews", fromlist=["run_pipeline"])
    run_pipeline = getattr(mod, "run_pipeline", None)
    assert callable(run_pipeline), "ingestion.finnews.run_pipeline manquant"

    items = run_pipeline(regions=["US"], window=1, query="AAPL", limit=10)
    assert isinstance(items, (list, tuple))
    # on tolère zéro si upstream vide, mais pas d'erreur
    if items:
        first = items[0]
        assert isinstance(first, (dict, object))


@pytest.mark.integration
def test_nlp_enrich_ask_model_integration():
    if not os.getenv("AF_ALLOW_INTERNET"):
        pytest.skip("Set AF_ALLOW_INTERNET=1 to run network integration tests")

    sys.path.insert(0, str(PROJECT_SRC))
    nlpm = __import__("research.nlp_enrich", fromlist=["ask_model"])  # type: ignore
    ask_model = getattr(nlpm, "ask_model", None)
    assert callable(ask_model), "research.nlp_enrich.ask_model manquant"
    out = ask_model("Ping?", context={"note": "integration-test"})
    assert out is not None
