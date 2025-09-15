#!/usr/bin/env python3
import os
import requests

# Simple test to check model connectivity
url = "http://127.0.0.1:4000/v1/chat/completions"

# Test different models
models_to_test = ["command-r", "aria", "flux"]

for model in models_to_test:
    print(f"Testing {model}...")
    try:
        response = requests.post(url, json={
            "model": model,
            "messages": [
                {"role": "user", "content": "Say hello"}
            ],
            "stream": False
        }, timeout=10)

        if response.status_code == 200:
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            print(f"  ✅ {model}: {content[:50]}...")
        else:
            print(f"  ❌ {model}: Status {response.status_code}")
            print(f"     {response.text[:100]}")

    except Exception as e:
        print(f"  ❌ {model}: Error - {e}")

print("\nDone testing models.")
