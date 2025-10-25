#!/usr/bin/env python3
from __future__ import annotations

import ast
import json
from collections import defaultdict, Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

LAYER_RULES = {
    "domain": ["core", "data"],
    "application": ["analytics", "runners", "research"],
    "adapters": ["ingestion", "apps", "utils", "mcp_server", "orch", "hub"],
}
LAYER_ORDER = {"domain": 0, "application": 1, "adapters": 2}


def module_name(path: Path) -> str:
    try:
        rel = path.relative_to(SRC).with_suffix("")
        parts = list(rel.parts)
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
        return ".".join(["src"] + parts)
    except Exception:
        return path.stem


def top_package(mod: str) -> str:
    parts = mod.split(".")
    if len(parts) >= 2 and parts[0] == "src":
        return parts[1]
    return parts[0]


def layer_of(mod: str) -> str:
    top = top_package(mod)
    for layer, pkgs in LAYER_RULES.items():
        if top in pkgs:
            return layer
    return "unknown"


def parse_imports(py: Path) -> Set[str]:
    try:
        tree = ast.parse(py.read_text(encoding="utf-8"))
    except Exception:
        return set()
    imps: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                imps.add(n.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imps.add(node.module)
    # normalize to src.* imports only
    norm: Set[str] = set()
    for m in imps:
        if m.startswith("src."):
            norm.add(m)
        else:
            # try to map to src.* if it's a top-level package of ours
            # skip stdlib/third-party
            pass
    return norm


def build_graph() -> Tuple[Dict[str, Set[str]], Dict[str, Dict[str, int]], Dict[str, int]]:
    graph: Dict[str, Set[str]] = defaultdict(set)
    files_by_mod: Dict[str, Dict[str, int]] = {}
    file_sizes: Dict[str, int] = {}
    for py in SRC.rglob("*.py"):
        if any(part.startswith(".") for part in py.parts):
            continue
        mod = module_name(py)
        file_sizes[str(py)] = py.stat().st_size
        imps = parse_imports(py)
        if mod not in files_by_mod:
            files_by_mod[mod] = {"files": 0, "imports": 0}
        files_by_mod[mod]["files"] += 1
        files_by_mod[mod]["imports"] += len(imps)
        for imp in imps:
            if imp != mod:
                graph[mod].add(imp)
    return graph, files_by_mod, file_sizes


def detect_cycles(graph: Dict[str, Set[str]]) -> List[List[str]]:
    cycles: List[List[str]] = []
    temp: Set[str] = set()
    perm: Set[str] = set()
    stack: List[str] = []

    def visit(n: str):
        if n in perm:
            return
        if n in temp:
            # found a cycle
            if n in stack:
                i = stack.index(n)
                cycles.append(stack[i:] + [n])
            return
        temp.add(n)
        stack.append(n)
        for m in graph.get(n, ()): visit(m)
        stack.pop()
        temp.remove(n)
        perm.add(n)

    for node in list(graph.keys()):
        visit(node)
    # deduplicate cycles by set signature
    seen = set()
    out = []
    for cyc in cycles:
        key = tuple(sorted(set(cyc)))
        if key not in seen:
            seen.add(key)
            out.append(cyc)
    return out


def layering_violations(graph: Dict[str, Set[str]]) -> List[Dict[str, str]]:
    vios = []
    for a, outs in graph.items():
        la = layer_of(a)
        ra = LAYER_ORDER.get(la, 99)
        for b in outs:
            lb = layer_of(b)
            rb = LAYER_ORDER.get(lb, 99)
            if rb < ra:
                vios.append({"from": a, "layer_from": la, "to": b, "layer_to": lb})
    return vios


def main():
    graph, files_meta, file_sizes = build_graph()
    cycles = detect_cycles(graph)
    vios = layering_violations(graph)

    # stats
    outdeg = {n: len(outs) for n, outs in graph.items()}
    indeg = Counter()
    for n, outs in graph.items():
        for m in outs:
            indeg[m] += 1
    hotspots = sorted(outdeg.items(), key=lambda x: x[1], reverse=True)[:20]
    heavy = sorted(file_sizes.items(), key=lambda x: x[1], reverse=True)[:20]

    analysis = {
        "summary": {
            "modules": len(graph),
            "edges": sum(len(v) for v in graph.values()),
            "cycles": len(cycles),
            "layer_violations": len(vios),
        },
        "hotspots": hotspots,
        "largest_files": heavy,
        "cycles": cycles,
        "layer_violations": vios,
    }

    outdir = ROOT / "docs" / "architecture"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")

    # brief markdown
    def line(s: str) -> str: return s + "\n"
    md = []
    md.append(line("# Architecture Analysis (static imports)"))
    md.append(line(f"- Modules: {analysis['summary']['modules']}, Edges: {analysis['summary']['edges']}"))
    md.append(line(f"- Cycles: {analysis['summary']['cycles']}, Layer violations: {analysis['summary']['layer_violations']}"))
    if cycles:
        md.append(line("\n## Cycles (sample)"))
        for cyc in cycles[:10]:
            md.append(line("- " + " → ".join(cyc)))
    if vios:
        md.append(line("\n## Layering Violations (from → to)"))
        for v in vios[:20]:
            md.append(line(f"- {v['from']} ({v['layer_from']}) → {v['to']} ({v['layer_to']})"))
    if hotspots:
        md.append(line("\n## Outgoing Dependencies (top)"))
        for n, d in hotspots:
            md.append(line(f"- {n}: {d}"))
    if heavy:
        md.append(line("\n## Largest Files (bytes)"))
        for p, sz in heavy:
            md.append(line(f"- {p}: {sz}"))
    (outdir / "analysis.md").write_text("".join(md), encoding="utf-8")

    print(json.dumps({"ok": True, "out": str(outdir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

