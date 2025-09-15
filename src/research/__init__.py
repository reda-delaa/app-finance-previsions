"""Research and experimental scripts package."""

# Expose NLP enrich functions for easy import
try:
    from .nlp_enrich import ask_model, enrich_article, sentiment_score, extract_entities
    __all__ = ["ask_model", "enrich_article", "sentiment_score", "extract_entities"]
except ImportError as e:
    print(f"Warning: Could not import NLP enrich functions: {e}")
    __all__ = []
