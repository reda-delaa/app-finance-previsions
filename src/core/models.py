"""
Core financial data structures and models.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any

import pandas as pd


@dataclass
class FinancialMetric:
    """Represents a single financial metric with value and metadata"""
    name: str
    value: float
    unit: str = "USD"
    period: str = "TTM"  # TTM, Q, Y
    date: Optional[datetime] = None
    source: Optional[str] = None
    confidence: float = 1.0

    def to_dict(self) -> Dict:
        """Convert to dictionary representation"""
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "period": self.period,
            "date": self.date.isoformat() if self.date else None,
            "source": self.source,
            "confidence": self.confidence
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'FinancialMetric':
        """Create from dictionary representation"""
        if 'date' in data and data['date']:
            data['date'] = datetime.fromisoformat(data['date'])
        return cls(**data)


@dataclass
class CompanyFinancials:
    """Container for company financial data and metrics"""
    symbol: str
    name: Optional[str] = None
    metrics: List[FinancialMetric] = None
    balance_sheet: Optional[pd.DataFrame] = None
    income_stmt: Optional[pd.DataFrame] = None
    cash_flow: Optional[pd.DataFrame] = None
    
    def __post_init__(self):
        """Initialize empty containers if None"""
        if self.metrics is None:
            self.metrics = []
    
    def add_metric(self, metric: FinancialMetric) -> None:
        """Add a financial metric"""
        self.metrics.append(metric)
    
    def get_metric(self, name: str, period: str = "TTM") -> Optional[FinancialMetric]:
        """Get most recent metric by name and period"""
        matching = [m for m in self.metrics if m.name == name and m.period == period]
        if not matching:
            return None
        return max(matching, key=lambda m: m.date or datetime.min)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation"""
        return {
            "symbol": self.symbol,
            "name": self.name,
            "metrics": [m.to_dict() for m in self.metrics],
            # DataFrames handled separately due to serialization
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CompanyFinancials':
        """Create from dictionary representation"""
        metrics = [FinancialMetric.from_dict(m) for m in data.get('metrics', [])]
        return cls(
            symbol=data['symbol'],
            name=data.get('name'),
            metrics=metrics
        )


class FinancialAnalysis:
    """Container for financial analysis results"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.company = CompanyFinancials(symbol)
        self.peers: List[CompanyFinancials] = []
        self.analysis_date = datetime.now()
        self.scores: Dict[str, float] = {}
        self.insights: List[Dict] = []
    
    def add_peer(self, peer: CompanyFinancials) -> None:
        """Add peer company analysis"""
        self.peers.append(peer)
    
    def add_score(self, name: str, score: float) -> None:
        """Add analysis score"""
        self.scores[name] = score
    
    def add_insight(self, category: str, message: str, 
                   confidence: float = 1.0, source: Optional[str] = None) -> None:
        """Add analysis insight"""
        self.insights.append({
            "category": category,
            "message": message,
            "confidence": confidence,
            "source": source,
            "timestamp": datetime.now().isoformat()
        })
    
    def to_dict(self) -> Dict:
        """Convert analysis to dictionary representation"""
        return {
            "symbol": self.symbol,
            "company": self.company.to_dict(),
            "peers": [p.to_dict() for p in self.peers],
            "analysis_date": self.analysis_date.isoformat(),
            "scores": self.scores,
            "insights": self.insights
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'FinancialAnalysis':
        """Create from dictionary representation"""
        analysis = cls(data['symbol'])
        analysis.company = CompanyFinancials.from_dict(data['company'])
        analysis.peers = [CompanyFinancials.from_dict(p) for p in data.get('peers', [])]
        analysis.analysis_date = datetime.fromisoformat(data['analysis_date'])
        analysis.scores = data.get('scores', {})
        analysis.insights = data.get('insights', [])
        return analysis


# ======================= Unified Features Bundle ========================
@dataclass
class FeatureBundle:
    """Unified features passed to IA/Arbitre.

    - macro: output of phase3_macro.get_macro_features() (dict-like)
    - technical: indicators/metrics for a ticker (dict-like)
    - fundamentals: basic financials/ratios (dict-like)
    - news: aggregated news signals (e.g., from analytics.market_intel.build_unified_features)
    - meta: optional context
    """
    macro: Optional[Dict[str, Any]] = None
    technical: Optional[Dict[str, Any]] = None
    fundamentals: Optional[Dict[str, Any]] = None
    news: Optional[Dict[str, Any]] = None
    meta: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self.macro is not None:
            out["macro"] = self._as_plain(self.macro)
        if self.technical is not None:
            out["technical"] = self._as_plain(self.technical)
        if self.fundamentals is not None:
            out["fundamentals"] = self._as_plain(self.fundamentals)
        if self.news is not None:
            out["news"] = self._as_plain(self.news)
        if self.meta is not None:
            out["meta"] = self._as_plain(self.meta)
        return out

    @staticmethod
    def _as_plain(obj: Any) -> Any:
        """Convert common containers to plain JSON-serializable structures."""
        try:
            if obj is None:
                return None
            if isinstance(obj, (str, int, float, bool)):
                return obj
            if isinstance(obj, dict):
                return {k: FeatureBundle._as_plain(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [FeatureBundle._as_plain(v) for v in obj]
            if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
                return FeatureBundle._as_plain(obj.to_dict())
            if hasattr(obj, "__dict__"):
                return FeatureBundle._as_plain(vars(obj))
        except Exception:
            pass
        return obj
