# MCP Setup pour orch-mcp

## Installation

pip install mcp fastmcp requests

## Configuration mcp.json

Ajoute cette entrée à ton fichier mcp.json dans VSCode/Cline :

```json
{
  "mcpServers": {
    "orch-mcp": {
      "command": "python",
      "args": ["-m", "mcp_server.main"],
      "env": {
        "API_BASE": "http://127.0.0.1:4000/v1",
        "MODEL": "command-r"
      }
    }
  }
}
```

## Lancement

Dans ton terminal depuis la racine du projet (nécessaire pour le PYTHONPATH) :
```bash
PYTHONPATH=$PWD python -m mcp_server.main
```

### Alternative (déconseillé) :
```bash
cd mcp_server && python main.py
```

## Intégration MCP dans VSCode

1. **Ouvre les settings de Cline** (`Cmd/Ctrl+Shift+P` → "Preferences: Open Settings")

2. **Ouvre ton fichier `mcp.json`** :
   - Windows/macOS : `~/Library/Application Support/Code/User/mcp.json` (ou `C:\Users\<username>\AppData\Roaming\Code\User\mcp.json`)

3. **Ajoute cette entrée** :
```json
{
  "mcpServers": {
    "orch-mcp": {
      "command": "python",
      "args": ["-m", "mcp_server.main"],
      "env": {
        "PYTHONPATH": "/Users/venom/Documents/analyse-financiere"
      }
    }
  }
}
```

4. **Relance VSCode/Cline** pour appliquer la configuration

### Outils MCP disponible dans Claude :
- `touch_file`: Créer un fichier vide
- `list_dir`: Lister contenu dossier
- `read_file`: Lire contenu fichier
- `write_file`: Écrire contenu fichier
- `shell`: Exécuter commande shell
