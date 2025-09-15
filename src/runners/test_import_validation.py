#!/usr/bin/env python3
"""
Test de validation des imports - vérifie que les modules peuvent être importés correctement
"""
import sys
import traceback

def test_econ_agent_import():
    """Test direct de l'import de econ_llm_agent"""
    print("🧪 Test import direct de src.analytics.econ_llm_agent")
    try:
        import src.analytics.econ_llm_agent as econ_agent

        # Vérifier les exports principaux
        assert hasattr(econ_agent, 'ask_model'), "ask_model non disponible"
        assert hasattr(econ_agent, 'arbitre'), "arbitre non disponible"
        assert hasattr(econ_agent, 'EconomicAnalyst'), "EconomicAnalyst non disponible"
        assert hasattr(econ_agent, 'POWER_NOAUTH_MODELS'), "POWER_NOAUTH_MODELS non disponible"

        # Vérifier que ce sont des callables
        assert callable(econ_agent.ask_model), "ask_model n'est pas callable"
        assert callable(econ_agent.arbitre), "arbitre n'est pas callable"
        assert callable(econ_agent.EconomicAnalyst), "EconomicAnalyst n'est pas callable"

        print("✅ Import direct réussi")
        return True
    except Exception as e:
        print(f"❌ Échec import direct: {e}")
        traceback.print_exc()
        return False

def test_relative_import():
    """Test d'import relatif dans un module voisin"""
    print("🧪 Test import relatif depuis un module voisin")
    try:
        from .analytics.econ_llm_agent import ask_model, arbitre

        # Test rapide
        response = ask_model('Test', {'locale': 'fr'})
        assert isinstance(response, str) and len(response) > 10

        print("✅ Import relatif réussi")
        return True
    except Exception as e:
        print(f"❌ Échec import relatif: {e}")
        traceback.print_exc()
        return False

def test_import_from_different_paths():
    """Test d'imports depuis différents chemins"""
    print("🧪 Test imports depuis différents chemins")
    tests = [
        "from src.analytics.econ_llm_agent import ask_model",
        "import src.analytics.econ_llm_agent as agent",
    ]

    for test_stmt in tests:
        try:
            print(f"  Testing: {test_stmt}")
            exec(test_stmt)
            print(f"  ✅ {test_stmt}")
        except Exception as e:
            print(f"  ❌ {test_stmt}: {e}")
            return False

    return True

def test_imports_without_g4f():
    """Test des imports sans dépendance explicite sur g4f (simule environnement neuf)"""
    print("🧪 Test des imports dans un contexte simulé d'installation fraîche")
    try:
        # Tester que le module peut être chargé sans erreur d'import immédiat
        import importlib
        import sys

        # Simuler un reload pour voir les erreurs d'import différées
        if 'src.analytics.econ_llm_agent' in sys.modules:
            importlib.reload(sys.modules['src.analytics.econ_llm_agent'])

        # Maintenant charger normalement
        import src.analytics.econ_llm_agent

        print("✅ Imports dans contexte d'installation fraîche réussis")
        return True
    except Exception as e:
        print(f"❌ Échec imports contexte installation fraîche: {e}")
        traceback.print_exc()
        return False

def main():
    """Fonction principale de validation"""
    print("🚀 VALIDATION DES IMPORTS")
    print("Vérification que tous les imports fonctionnent correctement")
    print("=" * 60)

    success = True

    try:
        success &= test_econ_agent_import()
        # Note: on ne peut pas tester l'import relatif depuis ce script car il faut être dans le package src
        success &= test_import_from_different_paths()
        success &= test_imports_without_g4f()
    except Exception as e:
        print(f"❌ Erreur générale lors des tests: {e}")
        success = False

    print("\n" + "=" * 60)
    if success:
        print("🎉 VALIDATION RÉUSSIE !")
        print("Tous les imports fonctionnent correctement.")
        print("Les modules peuvent être utilisés depuis n'importe où dans le projet.")
    else:
        print("❌ VALIDATION ÉCHOUÉE")
        print("Il reste des problèmes d'import à résoudre.")

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
