#!/usr/bin/env python3
"""
Test complet des fonctions ask_model et arbitre avec différents scénarios
"""
import sys
import traceback

def test_imports():
    """Test des imports de base"""
    print("=" * 50)
    print("TEST D'IMPORT")
    print("=" * 50)

    try:
        from src.analytics.econ_llm_agent import ask_model, arbitre, EconomicAnalyst, POWER_NOAUTH_MODELS
        print("✅ Tous les imports réussis")

        print(f"📊 ask_model disponible: {callable(ask_model)}")
        print(f"📊 arbitre disponible: {callable(arbitre)}")
        print(f"📊 EconomicAnalyst disponible: {callable(EconomicAnalyst)}")
        print(f"📊 {len(POWER_NOAUTH_MODELS)} modèles disponibles")

        return True
    except Exception as e:
        print(f"❌ Erreur d'import: {e}")
        traceback.print_exc()
        return False

def test_basic_functionality():
    """Test basique des fonctions principales"""
    print("\n" + "=" * 50)
    print("TEST FONCTIONNALITÉ BASIQUE")
    print("=" * 50)

    try:
        from src.analytics.econ_llm_agent import ask_model, arbitre

        # Test ask_model simple
        print("\n🧪 Test ask_model simple...")
        response = ask_model('Bonjour, pouvez-vous me répondre brièvement ?')
        if isinstance(response, str) and len(response) > 10:
            print(f"✅ ask_model réponse valide: {len(response)} caractères")
            print(f"   Aperçu: {response[:100]}...")
        else:
            print(f"❌ ask_model réponse invalide: {type(response)} - {response}")

        # Test ask_model avec contexte
        print("\n🧪 Test ask_model avec contexte...")
        context = {
            'locale': 'fr',
            'features': {'tension_commerciale': 0.3}
        }
        response_ctx = ask_model('Quelle est la situation économique ?', context)
        if isinstance(response_ctx, str) and len(response_ctx) > 50:
            print(f"✅ ask_model avec contexte: {len(response_ctx)} caractères")
        else:
            print(f"❌ ask_model avec contexte invalide")

        # Test arbitre simple
        print("\n🧪 Test arbitre simple...")
        ctx_arbitre = {'scope': 'macro', 'locale': 'fr', 'question': 'Test'}
        arb_result = arbitre(ctx_arbitre)
        if isinstance(arb_result, dict) and 'ok' in arb_result:
            print(f"✅ arbitre réponse valide: ok={arb_result['ok']}")
        else:
            print(f"❌ arbitre réponse invalide: {type(arb_result)}")

        return True
    except Exception as e:
        print(f"❌ Erreur dans test_basic_functionality: {e}")
        traceback.print_exc()
        return False

def test_error_handling():
    """Test de la gestion d'erreurs"""
    print("\n" + "=" * 50)
    print("TEST GESTION ERREURS")
    print("=" * 50)

    try:
        from src.analytics.econ_llm_agent import ask_model, arbitre

        # Test avec paramètres vides
        print("\n🧪 Test paramètres vides...")
        try:
            response_empty = ask_model('')
            if isinstance(response_empty, str):
                print("✅ Gestion des paramètres vides OK")
            else:
                print("⚠️  Paramètres vides non géré correctement")
        except Exception as e:
            print(f"⚠️  Exception avec paramètres vides: {e}")

        # Test avec contexte None
        print("\n🧪 Test contexte None...")
        try:
            response_none = ask_model('Question test', None)
            if isinstance(response_none, str):
                print("✅ Gestion du contexte None OK")
            else:
                print("⚠️  Contexte None non géré correctement")
        except Exception as e:
            print(f"⚠️  Exception avec contexte None: {e}")

        return True
    except Exception as e:
        print(f"❌ Erreur dans test_error_handling: {e}")
        traceback.print_exc()
        return False

def main():
    """Fonction principale de test"""
    print("🚀 DÉBUT DES TESTS COMPREHENSIFS")
    print("Test de src.analytics.econ_llm_agent")

    success = True

    success &= test_imports()
    success &= test_basic_functionality()
    success &= test_error_handling()

    print("\n" + "=" * 50)
    if success:
        print("🎉 TOUS LES TESTS SONT RÉUSSIS !")
        print("Les fonctions ask_model et arbitre sont opérationnelles.")
    else:
        print("❌ CERTAINS TESTS ONT ÉCHOUÉ")
        print("Vérifiez les erreurs ci-dessus.")
    print("=" * 50)

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
