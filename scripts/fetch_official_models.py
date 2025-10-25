"""
Fetch official g4f providers/models and write a local seed file:
data/llm/official/models.txt (format: provider|model or model)

If network is unavailable, writes a sample instructions file.
"""

from __future__ import annotations

from pathlib import Path
import sys


URL = 'https://g4f.dev/docs/providers-and-models'


def main() -> int:
    outdir = Path('data/llm/official')
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir/'models.txt'
    try:
        import requests
        r = requests.get(URL, timeout=20)
        r.raise_for_status()
        text = r.text
        # naive extraction: keep lines with separators or model‑looking tokens
        lines = []
        for raw in text.splitlines():
            s = raw.strip()
            if not s or len(s) > 180:
                continue
            if '|' in s and not s.startswith('<'):
                lines.append(s)
            # basic heuristic for model ids
            if any(k in s.lower() for k in ['llama','deepseek','qwen','glm','gpt-oss','mixtral','phi','gemma','qwq']) and '|' not in s and ' ' not in s and not s.startswith('<'):
                lines.append(s)
        if not lines:
            raise RuntimeError('no parsable lines')
        out.write_text('\n'.join(lines), encoding='utf-8')
        print({'ok': True, 'path': str(out), 'count': len(lines)})
        return 0
    except Exception as e:
        sample = (
            "# Could not fetch official list automatically.\n"
            "# Create lines like these:\n"
            "# provider|deepseek-ai/DeepSeek-R1-0528\n"
            "# provider|Qwen/Qwen3-235B-A22B-Thinking-2507\n"
            "# or just the model id per line if provider unknown:\n"
            "# deepseek-ai/DeepSeek-R1-0528\n"
            "# Qwen/Qwen3-235B-A22B-Thinking-2507\n"
        )
        out.write_text(sample, encoding='utf-8')
        print({'ok': False, 'error': str(e), 'path': str(out)})
        return 1


if __name__ == '__main__':
    sys.exit(main())

