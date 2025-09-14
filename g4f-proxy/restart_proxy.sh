#!/bin/bash

PORT=4000
SCRIPT="proxy.py"

echo "🔍 Recherche des processus sur le port $PORT..."
PIDS=$(lsof -ti :$PORT)

if [ -n "$PIDS" ]; then
    echo "🛑 Arrêt des processus : $PIDS"
    kill -9 $PIDS
else
    echo "✅ Aucun processus trouvé sur le port $PORT"
fi

echo "⏳ Attente 1 seconde..."
sleep 1

echo "🚀 Redémarrage de $SCRIPT..."
python "$SCRIPT"
