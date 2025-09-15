# g4f OpenAI-Compatible Proxy

Petit proxy qui expose un endpoint `/v1/chat/completions` compatible avec l’API OpenAI,
mais qui utilise **g4f** comme moteur LLM.

## Usage

1. Installer les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

2. Lancer le proxy :
   ```bash
   python proxy.py
   ```

3. Utiliser l’URL `http://localhost:4000/v1/chat/completions` comme `OPENAI_API_BASE` dans Cline.

## Tool calling

Si vous envoyez un payload contenant `tools: [...]`, le proxy ajoute une consigne
et tente de parser une réponse JSON de la forme :
```json
{ "name": "get_time", "arguments": { "tz": "UTC" } }
```

Cela sera renvoyé à Cline dans le format `tool_calls` OpenAI.
# Perfect non-stream request
curl -s http://127.0.0.1:4000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"command-r","messages":[{"role":"user","content":"Hello!"}]}' \
  | jq '.choices[0].message.content'

# Easy streaming  
./g4f-proxy/stream.sh aria "Give me a Python tip"

# Smoke test all working models
./g4f-proxy/smoke.sh

# Check which models work
curl -s http://127.0.0.1:4000/v1/working-models | jq '.ok[].variant'

