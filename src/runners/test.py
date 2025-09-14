# test_batch.py
import json
import logging
from src.research.peers_finder import get_peers_auto

# 20 tickers récents du S&P500 (fallback si Wikipédia marche pas)
tickers = [
    "PLTR","CRWD","ABNB","UBER","MRNA",
    "MELI","FTNT","DDOG","PANW","ZS",
    "ROKU","LI","ON","ANET","MDB",
    "TTD","CDNS","NOW","SNPS","FSLR",
]

def main():
    logging.basicConfig(level=logging.INFO)
    results = {}
    for t in tickers:
        print(f"\n=== Test {t} ===")
        try:
            peers = get_peers_auto(t, min_peers=5, max_peers=10, logger_=logging.getLogger("batch"))
            results[t] = peers
        except Exception as e:
            results[t] = {"error": str(e)}
    print("\n=== Résultats finaux ===")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()