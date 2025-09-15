# Test des fonctions ask_model et arbitre
try:
    from src.analytics.econ_llm_agent import ask_model, arbitre

    print(f'ask_model disponible: {ask_model is not None}')
    print(f'arbitre disponible: {arbitre is not None}')

    # Test simple des fonctions
    test_response = ask_model('Quelle est la situation économique ?')
    print('Réponse ask_model:', len(test_response), 'caractères')

    # Test arbitre avec un contexte basique
    test_ctx = {'scope': 'macro', 'locale': 'fr'}
    arb_response = arbitre(test_ctx)
    print('Réponse arbitre obtenue')

    print('\\n🎉 Toutes les corrections semblent fonctionner !')

except Exception as e:
    print(f'✗ Erreur: {e}')
    import traceback
    traceback.print_exc()
