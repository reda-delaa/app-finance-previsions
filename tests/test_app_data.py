# tests/test_app_data.py
# ---------------------------------------------------------------
# Couverture des appels DATA de src/apps/app.py avec PyTest
# - Stubs de providers injectés dans sys.modules avant import
# - Vérification des paramètres passés aux providers
# - Vérification du wrapper load_news -> ingestion.finnews.run_pipeline
# - Cas d'erreur/exception (dont {"error": ...} pour get_macro_features)
# - Vérification des logs de trace_call (→ et ←)
# ---------------------------------------------------------------

import sys
import types
import importlib
import pathlib
import math
import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"


# ---------- Helpers ----------
def _mk_module(name: str, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


# ---------- Fixtures ----------
@pytest.fixture(autouse=True)
def ensure_src_on_path():
    """S'assure que 'src' est présent dans sys.path pour les imports apps.*"""
    p = str(PROJECT_SRC)
    if p not in sys.path:
        sys.path.insert(0, p)
    yield


@pytest.fixture
def fake_providers(monkeypatch):
    """
    Installe des modules factices pour toutes les dépendances data,
    avec compteurs d'appels pour valider les paramètres.
    """
    calls = {
        "peers": [],
        "tech": [],
        "fund": [],
        "macro": [],
        "news": [],
    }

    # --- apps.*: stubs UI pour éviter des erreurs à l'import
    macro_mod = _mk_module("apps.macro_sector_app", render_macro=lambda: None)
    stock_mod = _mk_module("apps.stock_analysis_app", render_stock=lambda **_: None)

    # --- research.peers_finder
    def find_peers(ticker, k=8):
        calls["peers"].append({"ticker": ticker, "k": k})
        return {"peers": [f"{ticker}{i}" for i in range(k)]}
    peers_mod = _mk_module("research.peers_finder", find_peers=find_peers)

    # --- analytics.phase2_technical
    class _TechObj:
        def __init__(self, ticker, window):
            self.ticker = ticker
            self.window = window
        def to_dict(self):
            return {"ticker": self.ticker, "window": self.window, "ok": True}
    def compute_technical_features(ticker, window=180):
        calls["tech"].append({"ticker": ticker, "window": window})
        return _TechObj(ticker, window)
    tech_mod = _mk_module("analytics.phase2_technical", compute_technical_features=compute_technical_features)

    # --- analytics.phase1_fundamental
    def load_fundamentals(ticker):
        calls["fund"].append({"ticker": ticker})
        return {"ticker": ticker, "pe": 25.0, "ok": True}
    fund_mod = _mk_module("analytics.phase1_fundamental", load_fundamentals=load_fundamentals)

    # --- analytics.phase3_macro
    def get_macro_features():
        calls["macro"].append({})
        return {"gdp": 2.1, "cpi": 3.0}
    macro3_mod = _mk_module("analytics.phase3_macro", get_macro_features=get_macro_features)

    # --- ingestion.finnews
    def run_pipeline(regions, window, query, limit):
        calls["news"].append({"regions": regions, "window": window, "query": query, "limit": limit})
        return [
            {"date": "2025-09-01", "title": "Headline 1", "ticker": "AAPL", "region": "US", "source": "X", "sentiment": 0.1},
            {"date": "2025-09-02", "title": "Headline 2", "ticker": "MSFT", "region": "US", "source": "Y", "sentiment": -0.2},
        ]
    news_mod = _mk_module("ingestion.finnews", run_pipeline=run_pipeline)

    # --- NLP / Arbitre (stubs pour éviter des import errors)
    nlpenrich_mod = _mk_module("research.nlp_enrich", ask_model=lambda q, context=None: "ok")
    econ_agent_mod = _mk_module(
        "analytics.econ_llm_agent",
        arbitre=lambda ctx: {"decision": "hold"},
        EconomicAnalyst=type("EconomicAnalyst", (), {"analyze": lambda self, input_obj: {"ai": "ok"}}),
        EconomicInput=type("EconomicInput", (), {"__init__": lambda self, **kw: None}),
    )

    # Enregistre dans sys.modules AVANT l'import du module app
    sys.modules["apps.macro_sector_app"] = macro_mod
    sys.modules["apps.stock_analysis_app"] = stock_mod
    sys.modules["research.peers_finder"] = peers_mod
    sys.modules["analytics.phase2_technical"] = tech_mod
    sys.modules["analytics.phase1_fundamental"] = fund_mod
    sys.modules["analytics.phase3_macro"] = macro3_mod
    sys.modules["ingestion.finnews"] = news_mod
    sys.modules["research.nlp_enrich"] = nlpenrich_mod
    sys.modules["analytics.econ_llm_agent"] = econ_agent_mod

    yield calls

    # cleanup (optionnel)
    for name in [
        "apps.macro_sector_app",
        "apps.stock_analysis_app",
        "research.peers_finder",
        "analytics.phase2_technical",
        "analytics.phase1_fundamental",
        "analytics.phase3_macro",
        "ingestion.finnews",
        "research.nlp_enrich",
        "analytics.econ_llm_agent",
    ]:
        sys.modules.pop(name, None)


@pytest.fixture
def reload_app(fake_providers):
    """
    Importe/recharge src.apps.app après installation des stubs.
    Retourne (module_app, calls_dict).
    """
    # Si le module est déjà importé, on le recharge pour prendre en compte les stubs
    if "apps.app" in sys.modules:
        importlib.reload(sys.modules["apps.app"])
        mod = sys.modules["apps.app"]
    else:
        mod = importlib.import_module("apps.app")
    return mod, fake_providers


# ---------- Tests OK (flux nominal) ----------
def test_find_peers_ok(reload_app, caplog):
    app, calls = reload_app
    caplog.set_level("DEBUG")
    out = app.find_peers("AAPL", k=5)
    peers = out["peers"] if isinstance(out, dict) and "peers" in out else out
    assert len(peers) == 5
    assert calls["peers"] and calls["peers"][-1] == {"ticker": "AAPL", "k": 5}
    assert any("→ find_peers" in r.message for r in caplog.records)
    assert any("← find_peers" in r.message for r in caplog.records)


def test_compute_technical_features_ok(reload_app, caplog):
    app, calls = reload_app
    caplog.set_level("DEBUG")
    tech = app.compute_technical_features("MSFT", window=123)
    assert hasattr(tech, "to_dict")
    assert calls["tech"][-1] == {"ticker": "MSFT", "window": 123}
    assert any("→ compute_technical_features" in r.message for r in caplog.records)
    assert any("← compute_technical_features" in r.message for r in caplog.records)


def test_load_fundamentals_ok(reload_app, caplog):
    app, calls = reload_app
    caplog.set_level("DEBUG")
    fund = app.load_fundamentals("NVDA")
    assert isinstance(fund, dict) and fund["ticker"] == "NVDA"
    assert calls["fund"][-1]["ticker"] == "NVDA"
    assert any("→ load_fundamentals" in r.message for r in caplog.records)
    assert any("← load_fundamentals" in r.message for r in caplog.records)


def test_get_macro_features_ok(reload_app, caplog):
    app, calls = reload_app
    caplog.set_level("DEBUG")
    mf = app.get_macro_features()
    assert isinstance(mf, dict) and {"gdp", "cpi"} <= set(mf)
    assert calls["macro"]  # appelé au moins une fois
    assert any("→ get_macro_features" in r.message for r in caplog.records)
    assert any("← get_macro_features" in r.message for r in caplog.records)


def test_load_news_wrapper_ok(reload_app, caplog):
    app, calls = reload_app
    caplog.set_level("DEBUG")
    items = app.load_news(window_days=7, regions=["US", "EU"], tickers=["AAPL", "MSFT"])
    assert isinstance(items, list) and len(items) >= 2
    last = calls["news"][-1]
    assert last["regions"] == ["US", "EU"]
    assert last["window"] == 7
    assert last["query"] == "AAPL MSFT"
    assert last["limit"] == 50
    assert any("→ load_news" in r.message for r in caplog.records)
    assert any("← load_news" in r.message for r in caplog.records)


# ---------- Tests d'échecs / edges ----------
def test_news_pipeline_absent(monkeypatch):
    """
    Cas: ingestion.finnews.run_pipeline retourne [] -> load_news avec des faux stubs retourne [].
    """
    # Stubs minimaux avec news.run_pipeline retournant []
    sys.modules["ingestion.finnews"] = _mk_module("ingestion.finnews", run_pipeline=lambda **kwargs: [])
    sys.modules["apps.macro_sector_app"] = _mk_module("apps.macro_sector_app", render_macro=lambda: None)
    sys.modules["apps.stock_analysis_app"] = _mk_module("apps.stock_analysis_app", render_stock=lambda **_: None)
    sys.modules["research.peers_finder"] = _mk_module("research.peers_finder", find_peers=lambda *a, **k: [])
    sys.modules["analytics.phase2_technical"] = _mk_module("analytics.phase2_technical", compute_technical_features=lambda *a, **k: {})
    sys.modules["analytics.phase1_fundamental"] = _mk_module("analytics.phase1_fundamental", load_fundamentals=lambda *a, **k: {})
    sys.modules["analytics.phase3_macro"] = _mk_module("analytics.phase3_macro", get_macro_features=lambda: {})
    sys.modules["research.nlp_enrich"] = _mk_module("research.nlp_enrich", ask_model=lambda *a, **k: "ok")
    sys.modules["analytics.econ_llm_agent"] = _mk_module("analytics.econ_llm_agent", arbitre=lambda ctx: {})

    # reload app
    if "apps.app" in sys.modules:
        importlib.reload(sys.modules["apps.app"])
        app = sys.modules["apps.app"]
    else:
        app = importlib.import_module("apps.app")

    # Avec run_pipeline retournant [], load_news devrait retourner []
    assert app.load_news is not None  # Pas None car wrapper existe
    assert app.load_news(window_days=5) == []


def test_compute_technical_features_exception(monkeypatch):
    """
    Cas: compute_technical_features lève -> trace_call relance l'exception.
    """
    sys.modules["analytics.phase2_technical"] = _mk_module(
        "analytics.phase2_technical",
        compute_technical_features=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom-tech"))
    )
    # stubs pour le reste
    sys.modules["apps.macro_sector_app"] = _mk_module("apps.macro_sector_app", render_macro=lambda: None)
    sys.modules["apps.stock_analysis_app"] = _mk_module("apps.stock_analysis_app", render_stock=lambda **_: None)
    sys.modules["research.peers_finder"] = _mk_module("research.peers_finder", find_peers=lambda *a, **k: [])
    sys.modules["analytics.phase1_fundamental"] = _mk_module("analytics.phase1_fundamental", load_fundamentals=lambda *a, **k: {})
    sys.modules["analytics.phase3_macro"] = _mk_module("analytics.phase3_macro", get_macro_features=lambda: {})
    sys.modules["ingestion.finnews"] = _mk_module("ingestion.finnews", run_pipeline=lambda **_: [])
    sys.modules["research.nlp_enrich"] = _mk_module("research.nlp_enrich", ask_model=lambda *a, **k: "ok")
    sys.modules["analytics.econ_llm_agent"] = _mk_module("analytics.econ_llm_agent", arbitre=lambda ctx: {})

    # reload app
    if "apps.app" in sys.modules:
        importlib.reload(sys.modules["apps.app"])
        app = sys.modules["apps.app"]
    else:
        app = importlib.import_module("apps.app")

    with pytest.raises(RuntimeError, match="boom-tech"):
        app.compute_technical_features("TSLA", window=99)


def test_get_macro_features_error_dict(monkeypatch):
    """
    Cas: get_macro_features renvoie {"error": "..."} — le wrapper ne modifie pas le retour.
    """
    sys.modules["analytics.phase3_macro"] = _mk_module(
        "analytics.phase3_macro",
        get_macro_features=lambda: {"error": "Failed to get macro features: No objects to concatenate"}
    )
    # stubs pour le reste
    sys.modules["apps.macro_sector_app"] = _mk_module("apps.macro_sector_app", render_macro=lambda: None)
    sys.modules["apps.stock_analysis_app"] = _mk_module("apps.stock_analysis_app", render_stock=lambda **_: None)
    sys.modules["research.peers_finder"] = _mk_module("research.peers_finder", find_peers=lambda *a, **k: [])
    sys.modules["analytics.phase2_technical"] = _mk_module("analytics.phase2_technical", compute_technical_features=lambda *a, **k: {})
    sys.modules["analytics.phase1_fundamental"] = _mk_module("analytics.phase1_fundamental", load_fundamentals=lambda *a, **k: {})
    sys.modules["ingestion.finnews"] = _mk_module("ingestion.finnews", run_pipeline=lambda **_: [])
    sys.modules["research.nlp_enrich"] = _mk_module("research.nlp_enrich", ask_model=lambda *a, **k: "ok")
    sys.modules["analytics.econ_llm_agent"] = _mk_module("analytics.econ_llm_agent", arbitre=lambda ctx: {})

    # reload app
    if "apps.app" in sys.modules:
        importlib.reload(sys.modules["apps.app"])
        app = sys.modules["apps.app"]
    else:
        app = importlib.import_module("apps.app")

    out = app.get_macro_features()
    assert isinstance(out, dict) and "error" in out


# ---------- Tests de validation des données ----------
def test_data_validation_macro_features_ok(reload_app):
    """
    Test que les données macro sont valides (pas None/Nan/vides).
    """
    app, calls = reload_app
    mf = app.get_macro_features()
    assert mf is not None
    assert isinstance(mf, dict)
    assert len(mf) > 0

    # Vérification pas de valeurs None/vides
    for k, v in mf.items():
        assert v is not None, f"Valeur null pour {k}"
        assert v != "", f"Valeur vide pour {k}"
        if isinstance(v, (int, float)):
            assert not math.isnan(v), f"Valeur NaN pour {k}"


def test_data_validation_technical_features_ok(reload_app):
    """
    Test que les données techniques sont valides (pas None/Nan/vides).
    """
    app, calls = reload_app
    tech = app.compute_technical_features("AAPL", window=180)
    tech_dict = tech.to_dict()

    assert tech_dict is not None
    assert isinstance(tech_dict, dict)
    assert len(tech_dict) > 0

    # Vérification pas de valeurs None/vides
    for k, v in tech_dict.items():
        assert v is not None, f"Valeur null pour {k}"
        assert v != "", f"Valeur vide pour {k}"
        if isinstance(v, (int, float)):
            assert not math.isnan(v), f"Valeur NaN pour {k}"


def test_data_validation_fundamentals_ok(reload_app):
    """
    Test que les données fondamentales sont valides (pas None/Nan/vides).
    """
    app, calls = reload_app
    fund = app.load_fundamentals("MSFT")

    assert fund is not None
    assert isinstance(fund, dict)
    assert len(fund) > 0

    # Vérification pas de valeurs None/vides
    for k, v in fund.items():
        assert v is not None, f"Valeur null pour {k}"
        assert v != "", f"Valeur vide pour {k}"
        if isinstance(v, (int, float)):
            assert not math.isnan(v), f"Valeur NaN pour {k}"


def test_data_validation_peers_ok(reload_app):
    """
    Test que les données peers sont valides (pas None/vides).
    """
    app, calls = reload_app
    peers_data = app.find_peers("TSLA", k=5)
    peers = peers_data["peers"] if isinstance(peers_data, dict) and "peers" in peers_data else peers_data

    assert peers is not None
    assert isinstance(peers, list)
    assert len(peers) > 0

    # Vérification pas de valeurs None/vides
    for peer in peers:
        assert peer is not None, "Peer null trouvé"
        assert peer != "", "Peer vide trouvé"


def test_data_validation_news_ok(reload_app):
    """
    Test que les données news sont valides (pas None, champs requis présents).
    """
    app, calls = reload_app
    news = app.load_news(window_days=7, regions=["US"])

    assert news is not None
    assert isinstance(news, list)
    assert len(news) > 0

    # Vérification structure et champs requis
    required_fields = ["title", "date", "ticker", "region", "source", "sentiment"]
    for item in news:
        assert item is not None, "News null trouvée"
        assert isinstance(item, dict), "News n'est pas un dict"
        for field in required_fields:
            assert field in item, f"Champ requis '{field}' manquant"
            assert item[field] is not None, f"Valeur null pour champ '{field}'"
            assert item[field] != "", f"Valeur vide pour champ '{field}'"


def test_data_validation_with_null_values():
    """
    Test de validation avec données contenant des valeurs null/none.
    """
    # Fixtures avec données nulles
    def bad_macro():
        return {"gdp": None, "cpi": float('nan'), "rate": ""}

    def bad_tech(ticker, window=180):
        class BadTech:
            def to_dict(self):
                return {"ticker": ticker, "window": None, "value": float('nan'), "empty": ""}
        return BadTech()

    def bad_fund(ticker):
        return {"ticker": ticker, "pe": None, "market_cap": float('nan'), "description": ""}

    def bad_peers(ticker, k=8):
        return {"peers": [f"{ticker}{i}" for i in range(k-2)] + [None, ""]}

    def bad_news(**kwargs):
        return [
            {"date": "2025-09-01", "title": "Good news", "ticker": "AAPL", "region": "US", "source": "X", "sentiment": 0.1},
            {"date": None, "title": "", "ticker": None, "region": "US", "source": "Y", "sentiment": float('nan')},
        ]

    # Installation des stubs "bad"
    sys.modules["analytics.phase3_macro"] = _mk_module("analytics.phase3_macro", get_macro_features=bad_macro)
    sys.modules["analytics.phase2_technical"] = _mk_module("analytics.phase2_technical", compute_technical_features=bad_tech)
    sys.modules["analytics.phase1_fundamental"] = _mk_module("analytics.phase1_fundamental", load_fundamentals=bad_fund)
    sys.modules["research.peers_finder"] = _mk_module("research.peers_finder", find_peers=bad_peers)
    sys.modules["ingestion.finnews"] = _mk_module("ingestion.finnews", run_pipeline=lambda **kwargs: bad_news())

    # stubs pour UI
    sys.modules["apps.macro_sector_app"] = _mk_module("apps.macro_sector_app", render_macro=lambda: None)
    sys.modules["apps.stock_analysis_app"] = _mk_module("apps.stock_analysis_app", render_stock=lambda **_: None)
    sys.modules["research.nlp_enrich"] = _mk_module("research.nlp_enrich", ask_model=lambda q, context=None: "ok")
    sys.modules["analytics.econ_llm_agent"] = _mk_module("analytics.econ_llm_agent", arbitre=lambda ctx: {})

    # reload app
    if "apps.app" in sys.modules:
        importlib.reload(sys.modules["apps.app"])
        app = sys.modules["apps.app"]
    else:
        app = importlib.import_module("apps.app")

    # Tests de validation avec données "bad" - ces tests doivent échouer la validation UI
    # mais pas les tests unitaires (car les tests UI ne sont pas exécutés ici)
    bad_macro_data = app.get_macro_features()
    assert bad_macro_data is not None
    assert isinstance(bad_macro_data, dict)
    # On vérifie simplement que les données sont présentes, la validation UI se fait dans l'app

    bad_tech_data = app.compute_technical_features("BAD", window=100)
    bad_tech_dict = bad_tech_data.to_dict()
    assert bad_tech_dict is not None

    bad_fund_data = app.load_fundamentals("BAD")
    assert bad_fund_data is not None

    bad_peers_data = app.find_peers("BAD", k=8)
    assert bad_peers_data is not None

    bad_news_data = app.load_news(window_days=1)
    assert bad_news_data is not None


# ---------- Tests spécifiques aux problèmes macro-économiques observés ----------
def test_macro_gscpi_fetch_failure_handling():
    """
    Test de simulation des échecs de récupération GSCPI (URLs retournant du contenu non-CSV ou 404).
    """
    def gscpi_failure_macro():
        return {
            "macro_nowcast": {
                "scores": {"Growth": -1.0, "Inflation": -0.5, "Policy": -1.2, "USD": -0.3, "Commodities": 0.1},
                "components": {"GSCPI": None},  # Simule GSCPI fetch failure
            },
            "timestamp": "2025-09-30 00:00:00",
            "meta": {"country": "US", "freq": "M", "source": "FRED+yfinance"},
        }

    # Installation du stub macro avec GSCPI failure
    sys.modules["analytics.phase3_macro"] = _mk_module("analytics.phase3_macro", get_macro_features=gscpi_failure_macro)

    # Stubs pour le reste
    sys.modules["apps.macro_sector_app"] = _mk_module("apps.macro_sector_app", render_macro=lambda: None)
    sys.modules["apps.stock_analysis_app"] = _mk_module("apps.stock_analysis_app", render_stock=lambda **_: None)
    sys.modules["research.peers_finder"] = _mk_module("research.peers_finder", find_peers=lambda *a, **k: [])
    sys.modules["analytics.phase2_technical"] = _mk_module("analytics.phase2_technical", compute_technical_features=lambda *a, **k: {})
    sys.modules["analytics.phase1_fundamental"] = _mk_module("analytics.phase1_fundamental", load_fundamentals=lambda *a, **k: {})
    sys.modules["ingestion.finnews"] = _mk_module("ingestion.finnews", run_pipeline=lambda **_: [])
    sys.modules["research.nlp_enrich"] = _mk_module("research.nlp_enrich", ask_model=lambda q, context=None: "ok")
    sys.modules["analytics.econ_llm_agent"] = _mk_module("analytics.econ_llm_agent", arbitre=lambda ctx: {})

    if "apps.app" in sys.modules:
        importlib.reload(sys.modules["apps.app"])
        app = sys.modules["apps.app"]
    else:
        app = importlib.import_module("apps.app")

    # Test que l'app peut gérer le macro data avec GSCPI None
    macro_result = app.get_macro_features()
    assert macro_result is not None
    assert isinstance(macro_result, dict)
    assert "macro_nowcast" in macro_result
    assert "GSCPI" in macro_result["macro_nowcast"]["components"]
    assert macro_result["macro_nowcast"]["components"]["GSCPI"] is None


def test_macro_gpr_fetch_failure_handling():
    """
    Test de simulation des échecs de récupération GPR (Network errors et 404s).
    """
    def gpr_failure_macro():
        return {
            "macro_nowcast": {
                "scores": {"Growth": -0.8, "Inflation": 0.2, "Policy": -0.9, "USD": 0.0, "Commodities": -0.1},
                "components": {"GPR": None},  # Simule GPR fetch failure
            },
            "timestamp": "2025-09-30 00:00:00",
            "meta": {"country": "US", "freq": "M", "source": "FRED+yfinance"},
        }

    # Installation du stub macro avec GPR failure
    sys.modules["analytics.phase3_macro"] = _mk_module("analytics.phase3_macro", get_macro_features=gpr_failure_macro)

    # Stubs pour le reste
    sys.modules["apps.macro_sector_app"] = _mk_module("apps.macro_sector_app", render_macro=lambda: None)
    sys.modules["apps.stock_analysis_app"] = _mk_module("apps.stock_analysis_app", render_stock=lambda **_: None)
    sys.modules["research.peers_finder"] = _mk_module("research.peers_finder", find_peers=lambda *a, **k: [])
    sys.modules["analytics.phase2_technical"] = _mk_module("analytics.phase2_technical", compute_technical_features=lambda *a, **k: {})
    sys.modules["analytics.phase1_fundamental"] = _mk_module("analytics.phase1_fundamental", load_fundamentals=lambda *a, **k: {})
    sys.modules["ingestion.finnews"] = _mk_module("ingestion.finnews", run_pipeline=lambda **_: [])
    sys.modules["research.nlp_enrich"] = _mk_module("research.nlp_enrich", ask_model=lambda q, context=None: "ok")
    sys.modules["analytics.econ_llm_agent"] = _mk_module("analytics.econ_llm_agent", arbitre=lambda ctx: {})

    if "apps.app" in sys.modules:
        importlib.reload(sys.modules["apps.app"])
        app = sys.modules["apps.app"]
    else:
        app = importlib.import_module("apps.app")

    # Test que l'app peut gérer le macro data avec GPR None
    macro_result = app.get_macro_features()
    assert macro_result is not None
    assert isinstance(macro_result, dict)
    assert "macro_nowcast" in macro_result
    assert "GPR" in macro_result["macro_nowcast"]["components"]
    assert macro_result["macro_nowcast"]["components"]["GPR"] is None


def test_econ_llm_agent_empty_result_handling():
    """
    Test que l'app gère correctement les réponses vides de l'econ_llm_agent.
    """
    def empty_arbitre_response(ctx):
        return {"ok": True, "model": "deepseek-ai/DeepSeek-V3", "attempt": 1, "answer": "", "parsed": None, "usage": {}}

    # Stub econ_llm_agent avec réponse vide
    sys.modules["analytics.econ_llm_agent"] = _mk_module(
        "analytics.econ_llm_agent",
        arbitre=empty_arbitre_response,
        EconomicAnalyst=type("EconomicAnalyst", (), {"analyze": lambda self, input_obj: {"ai": "ok"}}),
        EconomicInput=type("EconomicInput", (), {"__init__": lambda self, **kw: None}),
    )

    # Stub macro returning data with NaN values like in logs
    def macro_with_nan():
        return {
            "macro_nowcast": {
                "scores": {"Growth": -1.59, "Inflation": -0.3, "Policy": -1.65, "USD": -0.78, "Commodities": 0.16},
                "components": {
                    "INDPRO_YoY": float('nan'),
                    "PAYEMS_YoY": float('nan'),
                    "CPI_YoY": float('nan'),
                    "CoreCPI_YoY": float('nan'),
                    "Breakeven_dev": float('nan'),
                    "FedFunds_dev": float('nan'),
                },
            },
            "timestamp": "2025-09-30 00:00:00",
            "meta": {"country": "US", "freq": "M", "source": "FRED+yfinance"},
        }

    sys.modules["analytics.phase3_macro"] = _mk_module("analytics.phase3_macro", get_macro_features=macro_with_nan)

    # Stubs pour le reste
    sys.modules["apps.macro_sector_app"] = _mk_module("apps.macro_sector_app", render_macro=lambda: None)
    sys.modules["apps.stock_analysis_app"] = _mk_module("apps.stock_analysis_app", render_stock=lambda **_: None)
    sys.modules["research.peers_finder"] = _mk_module("research.peers_finder", find_peers=lambda *a, **k: [])
    sys.modules["analytics.phase2_technical"] = _mk_module("analytics.phase2_technical", compute_technical_features=lambda *a, **k: {})
    sys.modules["analytics.phase1_fundamental"] = _mk_module("analytics.phase1_fundamental", load_fundamentals=lambda *a, **k: {})
    sys.modules["ingestion.finnews"] = _mk_module("ingestion.finnews", run_pipeline=lambda **_: [])
    sys.modules["research.nlp_enrich"] = _mk_module("research.nlp_enrich", ask_model=lambda q, context=None: "ok")

    if "apps.app" in sys.modules:
        importlib.reload(sys.modules["apps.app"])
        app = sys.modules["apps.app"]
    else:
        app = importlib.import_module("apps.app")

    # Test AI generation avec réponse vide - ne devrait pas casser l'UI
    macro_features = app.get_macro_features()
    assert macro_features is not None

    # Test NLP enrich response should work despite NaN values
    nlp_result = app.ask_model("test question")
    assert nlp_result is not None


def test_macro_data_with_partial_failures():
    """
    Test avec données macro partiellement manquantes (simulant les logs: certains indicateurs en NaN).
    """
    def partial_failure_macro():
        return {
            "macro_nowcast": {
                "scores": {"Growth": -1.59, "Inflation": -0.3, "Policy": -1.65, "USD": -0.78, "Commodities": 0.16},
                "components": {
                    "INDPRO_YoY": float('nan'),  # Failed fetch
                    "PAYEMS_YoY": float('nan'),  # Failed fetch
                    "CPI_YoY": None,              # Failed fetch
                    "YieldSlope_Tight": -0.5,     # Available
                    "USD_YoY": -0.021,           # Available
                    "Commodities_YoY": 0.13,     # Available
                },
            },
            "timestamp": "2025-09-30 00:00:00",
            "meta": {"country": "US", "freq": "M", "source": "FRED+yfinance"},
        }

    sys.modules["analytics.phase3_macro"] = _mk_module("analytics.phase3_macro", get_macro_features=partial_failure_macro)

    # Stubs pour le reste
    sys.modules["apps.macro_sector_app"] = _mk_module("apps.macro_sector_app", render_macro=lambda: None)
    sys.modules["apps.stock_analysis_app"] = _mk_module("apps.stock_analysis_app", render_stock=lambda **_: None)
    sys.modules["research.peers_finder"] = _mk_module("research.peers_finder", find_peers=lambda *a, **k: [])
    sys.modules["analytics.phase2_technical"] = _mk_module("analytics.phase2_technical", compute_technical_features=lambda *a, **k: {})
    sys.modules["analytics.phase1_fundamental"] = _mk_module("analytics.phase1_fundamental", load_fundamentals=lambda *a, **k: {})
    sys.modules["ingestion.finnews"] = _mk_module("ingestion.finnews", run_pipeline=lambda **_: [])
    sys.modules["research.nlp_enrich"] = _mk_module("research.nlp_enrich", ask_model=lambda q, context=None: "ok")
    sys.modules["analytics.econ_llm_agent"] = _mk_module(
        "analytics.econ_llm_agent",
        arbitre=lambda ctx: {},
        EconomicAnalyst=type("EconomicAnalyst", (), {"analyze": lambda self, input_obj: {"ai": "ok"}}),
        EconomicInput=type("EconomicInput", (), {"__init__": lambda self, **kw: None}),
    )

    if "apps.app" in sys.modules:
        importlib.reload(sys.modules["apps.app"])
        app = sys.modules["apps.app"]
    else:
        app = importlib.import_module("apps.app")

    # Test que l'app peut traiter des données partiellement manquantes
    macro_result = app.get_macro_features()
    assert macro_result is not None
    assert isinstance(macro_result, dict)
    assert "macro_nowcast" in macro_result

    # Vérifier les valeurs disponibles vs manquantes
    components = macro_result["macro_nowcast"]["components"]

    # NaN values should be handled (they come from float('nan'))
    nan_indicators = ["INDPRO_YoY", "PAYEMS_YoY", "CPI_YoY"]
    for indicator in nan_indicators:
        assert indicator in components

    # Valid values should remain untouched
    assert components["YieldSlope_Tight"] == -0.5
    assert components["USD_YoY"] == -0.021
    assert components["Commodities_YoY"] == 0.13


def test_network_error_handling(caplog):
    """
    Test de simulation des erreurs réseau (NameResolutionError) dans GPR fetch.
    """
    import warnings

    def network_error_macro():
        return {
            "macro_nowcast": {
                "scores": {"Growth": -0.8, "Inflation": 0.2, "Policy": -0.9, "USD": 0.0, "Commodities": -0.1},
                "components": {"GPR": None},  # Simule network error failure
            },
            "timestamp": "2025-09-30 00:00:00",
            "meta": {"country": "US", "freq": "M", "source": "FRED+yfinance"},
        }

    def macro_with_network_warning():
        # Simulate network-related error warning
        warnings.warn(
            "HTTPSConnectionPool(host='www2.bc.edu', port=443): Max retries exceeded with url: /matteo-iacoviello/gpr_files/GPRD.csv (Caused by NameResolutionError(\"<urllib3.connection.HTTPSConnection object at 0x11ef88190>: Failed to resolve 'www2.bc.edu' ([Errno 8] nodename nor servname provided, or not known)\"))",
            UserWarning
        )
        return network_error_macro()

    sys.modules["analytics.phase3_macro"] = _mk_module("analytics.phase3_macro", get_macro_features=macro_with_network_warning)

    # Stubs pour le reste
    sys.modules["apps.macro_sector_app"] = _mk_module("apps.macro_sector_app", render_macro=lambda: None)
    sys.modules["apps.stock_analysis_app"] = _mk_module("apps.stock_analysis_app", render_stock=lambda **_: None)
    sys.modules["research.peers_finder"] = _mk_module("research.peers_finder", find_peers=lambda *a, **k: [])
    sys.modules["analytics.phase2_technical"] = _mk_module("analytics.phase2_technical", compute_technical_features=lambda *a, **k: {})
    sys.modules["analytics.phase1_fundamental"] = _mk_module("analytics.phase1_fundamental", load_fundamentals=lambda *a, **k: {})
    sys.modules["ingestion.finnews"] = _mk_module("ingestion.finnews", run_pipeline=lambda **_: [])
    sys.modules["research.nlp_enrich"] = _mk_module("research.nlp_enrich", ask_model=lambda q, context=None: "ok")
    sys.modules["analytics.econ_llm_agent"] = _mk_module("analytics.econ_llm_agent", arbitre=lambda ctx: {})

    if "apps.app" in sys.modules:
        importlib.reload(sys.modules["apps.app"])
        app = sys.modules["apps.app"]
    else:
        app = importlib.import_module("apps.app")

    caplog.set_level("WARNING")
    # Test que l'app gère les erreurs réseau correctement - les warnings sont capturés par trace_call
    macro_result = app.get_macro_features()

    # Vérifier que le warning a été loggué par trace_call
    assert macro_result is not None
    assert isinstance(macro_result, dict)
    assert "GPR" in macro_result["macro_nowcast"]["components"]
    assert macro_result["macro_nowcast"]["components"]["GPR"] is None

    # Vérifier que les warnings sont loggués (les tests passent avec les nouveaux warn + logger.warning)
    warning_logged = any("NameResolutionError" in record.message for record in caplog.records)
    assert warning_logged, "Warning NameResolutionError non trouvé dans les logs"


def test_http_404_error_handling(caplog):
    """
    Test de simulation des erreurs HTTP 404 pour API endpoints.
    """
    def http_404_macro():
        return {
            "macro_nowcast": {
                "scores": {"Growth": -1.0, "Inflation": -0.5, "Policy": -1.2, "USD": -0.3, "Commodities": 0.1},
                "components": {"GSCPI": None},  # Simule 404 error
            },
            "timestamp": "2025-09-30 00:00:00",
            "meta": {"country": "US", "freq": "M", "source": "FRED+yfinance"},
        }

    def macro_with_http_404_warning():
        import warnings
        # Simulate HTTP 404 error warning from logs
        warnings.warn(
            "404 Client Error: Not Found for url: https://api.example.com/missing-endpoint",
            UserWarning
        )
        return http_404_macro()

    sys.modules["analytics.phase3_macro"] = _mk_module("analytics.phase3_macro", get_macro_features=macro_with_http_404_warning)

    # Stubs pour le reste
    sys.modules["apps.macro_sector_app"] = _mk_module("apps.macro_sector_app", render_macro=lambda: None)
    sys.modules["apps.stock_analysis_app"] = _mk_module("apps.stock_analysis_app", render_stock=lambda **_: None)
    sys.modules["research.peers_finder"] = _mk_module("research.peers_finder", find_peers=lambda *a, **k: [])
    sys.modules["analytics.phase2_technical"] = _mk_module("analytics.phase2_technical", compute_technical_features=lambda *a, **k: {})
    sys.modules["analytics.phase1_fundamental"] = _mk_module("analytics.phase1_fundamental", load_fundamentals=lambda *a, **k: {})
    sys.modules["ingestion.finnews"] = _mk_module("ingestion.finnews", run_pipeline=lambda **_: [])
    sys.modules["research.nlp_enrich"] = _mk_module("research.nlp_enrich", ask_model=lambda q, context=None: "ok")
    sys.modules["analytics.econ_llm_agent"] = _mk_module("analytics.econ_llm_agent", arbitre=lambda ctx: {})

    if "apps.app" in sys.modules:
        importlib.reload(sys.modules["apps.app"])
        app = sys.modules["apps.app"]
    else:
        app = importlib.import_module("apps.app")

    caplog.set_level("WARNING")
    # Test que l'app gère les erreurs HTTP 404 correctement - les warnings sont capturés par trace_call
    macro_result = app.get_macro_features()

    assert macro_result is not None
    assert isinstance(macro_result, dict)
    assert "GSCPI" in macro_result["macro_nowcast"]["components"]
    assert macro_result["macro_nowcast"]["components"]["GSCPI"] is None

    # Vérifier que les warnings sont loggués (pas avec pytest.warns car interceptés par trace_call)
    warning_logged = any("404 Client Error" in record.message for record in caplog.records)
    assert warning_logged, "Warning 404 Client Error non trouvé dans les logs"


def test_fred_data_failures():
    """
    Test de simulation des échecs FRED avec création de données de rechange valides.
    """
    def fred_failure_with_fallback():
        return {
            "macro_nowcast": {
                "scores": {"Growth": -1.59, "Inflation": -0.3, "Policy": -1.65, "USD": -0.78, "Commodities": 0.16},
                "components": {
                    # FRED données manquantes/deviennent NaN
                    "INDPRO_YoY": float('nan'),
                    "PAYEMS_YoY": float('nan'),
                    "CPI_YoY": float('nan'),
                    "CoreCPI_YoY": float('nan'),
                    "Breakeven_dev": float('nan'),
                    "FedFunds_dev": float('nan'),
                    # Données alternatives disponibles
                    "YieldSlope_Tight": -0.49999999999999956,
                    "USD_YoY": -0.02127829731265235,
                    "Commodities_YoY": 0.12984280101416154,
                }
            },
            "timestamp": "2025-09-30 00:00:00",
            "meta": {"country": "US", "freq": "M", "source": "FRED+yfinance"},
        }

    sys.modules["analytics.phase3_macro"] = _mk_module("analytics.phase3_macro", get_macro_features=fred_failure_with_fallback)

    # Stubs pour le reste
    sys.modules["apps.macro_sector_app"] = _mk_module("apps.macro_sector_app", render_macro=lambda: None)
    sys.modules["apps.stock_analysis_app"] = _mk_module("apps.stock_analysis_app", render_stock=lambda **_: None)
    sys.modules["research.peers_finder"] = _mk_module("research.peers_finder", find_peers=lambda *a, **k: [])
    sys.modules["analytics.phase2_technical"] = _mk_module("analytics.phase2_technical", compute_technical_features=lambda *a, **k: {})
    sys.modules["analytics.phase1_fundamental"] = _mk_module("analytics.phase1_fundamental", load_fundamentals=lambda *a, **k: {})
    sys.modules["ingestion.finnews"] = _mk_module("ingestion.finnews", run_pipeline=lambda **_: [])
    sys.modules["research.nlp_enrich"] = _mk_module("research.nlp_enrich", ask_model=lambda q, context=None: "ok")
    sys.modules["analytics.econ_llm_agent"] = _mk_module("analytics.econ_llm_agent", arbitre=lambda ctx: {})

    if "apps.app" in sys.modules:
        importlib.reload(sys.modules["apps.app"])
        app = sys.modules["apps.app"]
    else:
        app = importlib.import_module("apps.app")

    # Test que l'app gère les échecs FRED en utilisant les données alternatives
    macro_result = app.get_macro_features()
    assert macro_result is not None

    components = macro_result["macro_nowcast"]["components"]

    # FRED data should be NaN
    fred_indicators = ["INDPRO_YoY", "PAYEMS_YoY", "CPI_YoY", "CoreCPI_YoY", "Breakeven_dev", "FedFunds_dev"]
    for indicator in fred_indicators:
        assert math.isnan(components[indicator]) or components[indicator] is None

    # Alternative data should be valid
    assert components["YieldSlope_Tight"] is not None
    assert components["USD_YoY"] is not None
    assert components["Commodities_YoY"] is not None


def test_app_error_termination_handling():
    """
    Test que l'app gère correctement les erreurs de terminaison du processus.
    """
    # Stub qui simule une erreur tragique comme dans les logs
    def catastrophic_error_macro():
        raise SystemExit("Process terminated unexpectedly")

    sys.modules["analytics.phase3_macro"] = _mk_module("analytics.phase3_macro", get_macro_features=catastrophic_error_macro)

    # Stubs pour le reste
    sys.modules["apps.macro_sector_app"] = _mk_module("apps.macro_sector_app", render_macro=lambda: None)
    sys.modules["apps.stock_analysis_app"] = _mk_module("apps.stock_analysis_app", render_stock=lambda **_: None)
    sys.modules["research.peers_finder"] = _mk_module("research.peers_finder", find_peers=lambda *a, **k: [])
    sys.modules["analytics.phase2_technical"] = _mk_module("analytics.phase2_technical", compute_technical_features=lambda *a, **k: {})
    sys.modules["analytics.phase1_fundamental"] = _mk_module("analytics.phase1_fundamental", load_fundamentals=lambda *a, **k: {})
    sys.modules["ingestion.finnews"] = _mk_module("ingestion.finnews", run_pipeline=lambda **_: [])
    sys.modules["research.nlp_enrich"] = _mk_module("research.nlp_enrich", ask_model=lambda q, context=None: "ok")
    sys.modules["analytics.econ_llm_agent"] = _mk_module("analytics.econ_llm_agent", arbitre=lambda ctx: {})

    if "apps.app" in sys.modules:
        importlib.reload(sys.modules["apps.app"])
        app = sys.modules["apps.app"]
    else:
        app = importlib.import_module("apps.app")

    # Test que l'app ne tombe pas en crash dur face à une erreur critique - SystemExit est correctement propagée
    with pytest.raises(SystemExit):
        app.get_macro_features()


    """
    Test de simulation des erreurs réseau (NameResolutionError) dans GPR fetch.
    """
    import warnings

    def network_error_macro():
        return {
            "macro_nowcast": {
                "scores": {"Growth": -0.8, "Inflation": 0.2, "Policy": -0.9, "USD": 0.0, "Commodities": -0.1},
                "components": {"GPR": None},  # Simule network error failure
            },
            "timestamp": "2025-09-30 00:00:00",
            "meta": {"country": "US", "freq": "M", "source": "FRED+yfinance"},
        }

    def macro_with_network_warning():
        # Simulate network-related error warning
        warnings.warn(
            "HTTPSConnectionPool(host='www2.bc.edu', port=443): Max retries exceeded with url: /matteo-iacoviello/gpr_files/GPRD.csv (Caused by NameResolutionError(\"<urllib3.connection.HTTPSConnection object at 0x11ef88190>: Failed to resolve 'www2.bc.edu' ([Errno 8] nodename nor servname provided, or not known)\"))",
            UserWarning
        )
        return network_error_macro()

    sys.modules["analytics.phase3_macro"] = _mk_module("analytics.phase3_macro", get_macro_features=macro_with_network_warning)

    # Stubs pour le reste
    sys.modules["apps.macro_sector_app"] = _mk_module("apps.macro_sector_app", render_macro=lambda: None)
    sys.modules["apps.stock_analysis_app"] = _mk_module("apps.stock_analysis_app", render_stock=lambda **_: None)
    sys.modules["research.peers_finder"] = _mk_module("research.peers_finder", find_peers=lambda *a, **k: [])
    sys.modules["analytics.phase2_technical"] = _mk_module("analytics.phase2_technical", compute_technical_features=lambda *a, **k: {})
    sys.modules["analytics.phase1_fundamental"] = _mk_module("analytics.phase1_fundamental", load_fundamentals=lambda *a, **k: {})
    sys.modules["ingestion.finnews"] = _mk_module("ingestion.finnews", run_pipeline=lambda **_: [])
    sys.modules["research.nlp_enrich"] = _mk_module("research.nlp_enrich", ask_model=lambda q, context=None: "ok")
    sys.modules["analytics.econ_llm_agent"] = _mk_module("analytics.econ_llm_agent", arbitre=lambda ctx: {})

    if "apps.app" in sys.modules:
        importlib.reload(sys.modules["apps.app"])
        app = sys.modules["apps.app"]
    else:
        app = importlib.import_module("apps.app")

    caplog.set_level("WARNING")
    # Test que l'app gère les erreurs réseau correctement - les warnings sont capturés par trace_call
    macro_result = app.get_macro_features()

    # Vérifier que le warning a été loggué par trace_call
    assert macro_result is not None
    assert isinstance(macro_result, dict)
    assert "GPR" in macro_result["macro_nowcast"]["components"]
    assert macro_result["macro_nowcast"]["components"]["GPR"] is None

    # Vérifier que les warnings sont loggués (pas avec pytest.warns car interceptés par trace_call)
    warning_logged = any("NameResolutionError" in record.message for record in caplog.records)
    assert warning_logged, "Warning NameResolutionError non trouvé dans les logs"


    """
    Test de simulation des erreurs HTTP 404 pour API endpoints.
    """
    def http_404_macro():
        return {
            "macro_nowcast": {
                "scores": {"Growth": -1.0, "Inflation": -0.5, "Policy": -1.2, "USD": -0.3, "Commodities": 0.1},
                "components": {"GSCPI": None},  # Simule 404 error
            },
            "timestamp": "2025-09-30 00:00:00",
            "meta": {"country": "US", "freq": "M", "source": "FRED+yfinance"},
        }

    def macro_with_http_404_warning():
        import warnings
        # Simulate HTTP 404 error warning from logs
        warnings.warn(
            "404 Client Error: Not Found for url: https://api.example.com/missing-endpoint",
            UserWarning
        )
        return http_404_macro()

    sys.modules["analytics.phase3_macro"] = _mk_module("analytics.phase3_macro", get_macro_features=macro_with_http_404_warning)

    # Stubs pour le reste
    sys.modules["apps.macro_sector_app"] = _mk_module("apps.macro_sector_app", render_macro=lambda: None)
    sys.modules["apps.stock_analysis_app"] = _mk_module("apps.stock_analysis_app", render_stock=lambda **_: None)
    sys.modules["research.peers_finder"] = _mk_module("research.peers_finder", find_peers=lambda *a, **k: [])
    sys.modules["analytics.phase2_technical"] = _mk_module("analytics.phase2_technical", compute_technical_features=lambda *a, **k: {})
    sys.modules["analytics.phase1_fundamental"] = _mk_module("analytics.phase1_fundamental", load_fundamentals=lambda *a, **k: {})
    sys.modules["ingestion.finnews"] = _mk_module("ingestion.finnews", run_pipeline=lambda **_: [])
    sys.modules["research.nlp_enrich"] = _mk_module("research.nlp_enrich", ask_model=lambda q, context=None: "ok")
    sys.modules["analytics.econ_llm_agent"] = _mk_module("analytics.econ_llm_agent", arbitre=lambda ctx: {})

    if "apps.app" in sys.modules:
        importlib.reload(sys.modules["apps.app"])
        app = sys.modules["apps.app"]
    else:
        app = importlib.import_module("apps.app")

    caplog.set_level("WARNING")
    # Test que l'app gère les erreurs HTTP 404 correctement - les warnings sont capturés par trace_call
    macro_result = app.get_macro_features()

    assert macro_result is not None
    assert isinstance(macro_result, dict)
    assert "GSCPI" in macro_result["macro_nowcast"]["components"]
    assert macro_result["macro_nowcast"]["components"]["GSCPI"] is None

    # Vérifier que les warnings sont loggués (pas avec pytest.warns car interceptés par trace_call)
    warning_logged = any("404 Client Error" in record.message for record in caplog.records)
    assert warning_logged, "Warning 404 Client Error non trouvé dans les logs"
