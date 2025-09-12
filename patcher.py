#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
Applique un patch texte sur un fichier cible, sans dépendances externes.

• Supporte :
  - Unified diff classique:  --- a/path   +++ b/path   @@ -l,c +l,c @@
  - "Unified-ish" minimal:   @@ ... @@ avec lignes '-' '+' et contexte neutre

• Tolérant :
  - Nettoie le patch (enlève ``` fences, '%' résiduel, BOM, CRLF)
  - Auto-détection des fichiers ciblés et strip des préfixes a/ b/
  - Matching exact → normalisé espaces → fuzzy regex (\s+)
  - Insertion guidée par le CONTEXTE si OLD est vide
  - Repli "bloc-fonction": si OLD contient `def foo(...):` ou `class Bar:`,
    remplacement du bloc # col.0 → prochain def/class col.0 (ou EOF)
  - Repli affectation: si OLD/NEW ressemble à `VAR = ...`, on remplace la ligne d’affectation
    même si le commentaire diffère.
  - Repli ultra-tolérant pour petits hunks inline (quotes/ponctuation/espaces)

• Usage :
  python patcher.py fix.patch path/to/target.py [--check] [--backup <path>]

Limitations :
  - Pas de détection/creation automatique si le patch cible un autre fichier que "target".
  - Les réarrangements massifs ou conflits sérieux nécessitent une revue manuelle.
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path
import sys
from typing import List, Tuple, Optional

# ----------------- Sanitize helpers -----------------

def sanitize_patch_text(raw: str) -> str:
    """Nettoie le patch : enlève BOM/CRLF/fences/%, assure newline final."""
    lines = [L for L in raw.splitlines() if not L.strip().startswith("```")]
    txt = "\n".join(lines)
    txt = txt.replace("\r\n", "\n").replace("\r", "\n")
    txt = txt.lstrip("\ufeff")
    txt = txt.rstrip()
    if txt.endswith("%") and "\n" not in (txt.splitlines()[-1] if txt else ""):
        txt = txt[:-1].rstrip()
    if not txt.endswith("\n"):
        txt += "\n"
    return txt

# ----------------- Parse helpers -----------------

HDR_RE = re.compile(r"^---\s+(.+)\n\+\+\+\s+(.+)\n", re.M)

def split_file_sections(txt: str) -> List[Tuple[Optional[str], str]]:
    m = list(HDR_RE.finditer(txt))
    if not m:
        return [(None, txt)]
    out: List[Tuple[Optional[str], str]] = []
    indices = [mm.start() for mm in m] + [len(txt)]
    for i, mm in enumerate(m):
        start = mm.start()
        end = indices[i+1]
        block = txt[start:end]
        newp = mm.group(2).strip()
        def _strip_ab(p: str) -> str:
            if p.startswith("a/") or p.startswith("b/"):
                return p[2:]
            return p
        path = _strip_ab(newp)
        out.append((path, block))
    return out

def split_hunks(section_text: str) -> List[str]:
    parts: List[str] = []
    current: List[str] = []
    saw_at = False
    for line in section_text.splitlines(keepends=True):
        if line.startswith("@@"):
            if current:
                parts.append("".join(current))
                current = []
            saw_at = True
        current.append(line)
    if current and saw_at:
        parts.append("".join(current))
    if not parts:
        return [section_text]
    return parts

# ----------------- Hunk extraction -----------------

def extract_old_new(hunk: str) -> Tuple[str, str, str]:
    old_lines: List[str] = []
    new_lines: List[str] = []
    ctx_lines: List[str] = []
    for line in hunk.splitlines(keepends=True):
        if line.startswith("@@") or line.startswith("---") or line.startswith("+++"):
            continue
        if line.startswith("-"):
            old_lines.append(line[1:])
        elif line.startswith("+"):
            new_lines.append(line[1:])
        else:
            ctx_lines.append(line[1:] if line and line[0] in " " else line)
    return ("".join(old_lines), "".join(new_lines), "".join(ctx_lines))

# ----------------- Matching helpers -----------------

def _normalize_ws(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"[ \t]+\n", "\n", s)
    return s

def _spacey_pattern(s: str) -> str:
    s = re.escape(s)
    s = s.replace(r"\ ", r"\s+")
    return s

def _strip_comment(s: str) -> str:
    """Supprime les commentaires Python (#...) pour comparer du code."""
    return re.sub(r"#.*", "", s).rstrip()

# --- fonction / classe detection ---

def _extract_symbol_name(block: str) -> Optional[Tuple[str, Optional[str]]]:
    for line in block.splitlines():
        if line.startswith("def ") or line.startswith("class "):
            kind = "def" if line.startswith("def ") else "class"
            m = re.match(rf"{kind}\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", line)
            if m:
                return (kind, m.group(1))
            if kind == "class":
                m2 = re.match(r"class\s+([A-Za-z_][A-Za-z0-9_]*)\s*:", line)
                if m2:
                    return ("class", m2.group(1))
            return (kind, None)
    return None

def _find_symbol_block(hay: str, kind: str, name: Optional[str]) -> Optional[Tuple[int, int]]:
    lines = hay.splitlines(keepends=True)
    start = None
    pat = re.compile(rf"^{kind}\s+{re.escape(name)}\b") if name else re.compile(rf"^{kind}\s+")
    for i, l in enumerate(lines):
        if pat.match(l):
            start = i
            break
    if start is None:
        return None
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if lines[j].startswith("def ") or lines[j].startswith("class "):
            end = j
            break
    a = sum(len(x) for x in lines[:start])
    b = sum(len(x) for x in lines[:end])
    return (a, b)

# --- affectations ---

_ASSIGN_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*.+$", re.S)

def _assignment_var(s: str) -> Optional[str]:
    if "\n" in s:
        return None
    m = _ASSIGN_RE.match(_strip_comment(s).strip())
    return m.group(1) if m else None

def _replace_assignment_line(hay: str, var: str, new_line: str) -> Tuple[str, bool]:
    pat = re.compile(rf"(?m)^[ \t]*{re.escape(var)}\s*=.*$")
    m = pat.search(hay)
    if not m:
        return hay, False
    start, end = m.span()
    repl = new_line.rstrip("\n")
    return hay[:start] + repl + hay[end:], True

# ----------------- Core replacement -----------------

def replace_once_with_fallbacks(hay: str, old: str, new: str, ctx: str) -> Tuple[str, bool]:
    # 0) idempotence (ignore commentaires)
    if new:
        Hn = _normalize_ws(_strip_comment(hay))
        Nn = _normalize_ws(_strip_comment(new))
        if Nn and (Nn in Hn):
            return (hay, True)

    # insertion pure
    if not old and new:
        if ctx.strip():
            ctx_lines = [l for l in ctx.splitlines(keepends=True) if l.strip()]
            anchor = ctx_lines[-1] if ctx_lines else None
            if anchor:
                loc = hay.find(anchor)
                if loc >= 0:
                    insert_at = loc + len(anchor)
                    return (hay[:insert_at] + new + hay[insert_at:], True)
        return (hay + ("" if hay.endswith("\n") else "\n") + new, True)

    # exact
    i = hay.find(old)
    if i >= 0:
        return (hay[:i] + new + hay[i+len(old):], True)

    # normalisé
    H = _normalize_ws(hay)
    O = _normalize_ws(old)
    j = H.find(O)
    if j >= 0:
        pat = _spacey_pattern(old)
        m = re.search(pat, hay, flags=re.DOTALL)
        if m:
            a, b = m.span()
            return (hay[:a] + new + hay[b:], True)

    # regex direct
    pat = _spacey_pattern(old)
    m = re.search(pat, hay, flags=re.DOTALL)
    if m:
        a, b = m.span()
        return (hay[:a] + new + hay[b:], True)

    # bloc fonction
    sym = _extract_symbol_name(old)
    if sym:
        kind, name = sym
        loc = _find_symbol_block(hay, kind, name)
        if loc:
            a, b = loc
            return (hay[:a] + new + hay[b:], True)

    # affectation
    var = _assignment_var(old) or _assignment_var(new)
    if var:
        out, ok = _replace_assignment_line(hay, var, new.strip("\n"))
        if ok:
            return out, True

    # ultra tolérant
    if len(old) <= 120:
        def _ultra(s: str) -> str:
            s = s.lower()
            s = re.sub(r"[\"'`]", "", s)
            s = re.sub(r"[^a-z0-9]+", " ", s)
            s = re.sub(r"\s+", " ", s).strip()
            return s
        Uhay = _ultra(hay)
        Uold = _ultra(old)
        if Uold and (Uold in Uhay):
            pat = _spacey_pattern(old)
            m2 = re.search(pat, hay, flags=re.DOTALL | re.IGNORECASE)
            if m2:
                a, b = m2.span()
                return (hay[:a] + new + hay[b:], True)

    return (hay, False)

# ----------------- Core apply -----------------

def apply_patch_to_text(patch_text: str, target_text: str, target_path_hint: Optional[str] = None) -> Tuple[str, List[str], int, int]:
    errors: List[str] = []
    patch_text = sanitize_patch_text(patch_text)
    sections = split_file_sections(patch_text)
    filtered: List[Tuple[Optional[str], str]] = []
    if all(p is None for p, _ in sections):
        filtered = sections
    else:
        tgt = (target_path_hint or "").replace("\\", "/")
        tgt_tail = tgt.split("/")[-1] if tgt else ""
        for pth, sec in sections:
            if pth is None:
                continue
            p = pth.replace("\\", "/")
            if p == tgt or p.endswith("/" + tgt_tail) or p.endswith(tgt_tail):
                filtered.append((pth, sec))
        if not filtered:
            filtered = sections

    out = target_text
    hunks_total = 0
    hunks_ok = 0
    for _, sec in filtered:
        hunks = split_hunks(sec)
        for h in hunks:
            hunks_total += 1
            old, new, ctx = extract_old_new(h)
            if not old and not new:
                continue
            out2, ok = replace_once_with_fallbacks(out, old, new, ctx)
            if ok:
                out = out2
                hunks_ok += 1
            else:
                preview = (old[:80] if old else "").replace("\n", "\\n")
                errors.append(f"hunk #{hunks_total} : bloc OLD introuvable (len={len(old)} ; preview='{preview}')")

    return out, errors, hunks_ok, hunks_total

# ----------------- CLI -----------------

def main():
    ap = argparse.ArgumentParser(description="Applique un patch (diff-like) à un fichier cible, sans git.")
    ap.add_argument("patch_file", help="Chemin vers fix.patch")
    ap.add_argument("target_file", help="Fichier cible à modifier")
    ap.add_argument("--backup", help="Chemin du backup (défaut: <target>.bak)")
    ap.add_argument("--check", action="store_true", help="Dry-run : n’écrit rien, affiche seulement le rapport")
    args = ap.parse_args()

    p_patch = Path(args.patch_file)
    p_tgt = Path(args.target_file)

    if not p_patch.exists():
        print(f"[ERR] Patch introuvable: {p_patch}", file=sys.stderr); sys.exit(1)
    if not p_tgt.exists():
        print(f"[ERR] Fichier cible introuvable: {p_tgt}", file=sys.stderr); sys.exit(1)

    patch_text = p_patch.read_text(encoding="utf-8", errors="replace")
    target_text = p_tgt.read_text(encoding="utf-8", errors="replace")

    new_text, errors, ok, total = apply_patch_to_text(patch_text, target_text, target_path_hint=str(p_tgt))

    if args.check:
        if errors:
            print(f"[CHECK] {ok}/{total} hunks matchés. Échecs :")
            for e in errors: print("  -", e)
            sys.exit(2)
        else:
            print(f"[CHECK] Tous les hunks ({total}) seraient appliqués.")
            sys.exit(0)

    backup_path = Path(args.backup) if args.backup else p_tgt.with_suffix(p_tgt.suffix + ".bak")
    backup_path.write_text(target_text, encoding="utf-8")
    p_tgt.write_text(new_text, encoding="utf-8")

    if errors:
        print(f"[OK avec avertissements] {ok}/{total} hunks appliqués. Backup: {backup_path}")
        for e in errors: print("  -", e)
        sys.exit(0)
    else:
        print(f"[OK] {ok}/{total} hunks appliqués. Backup: {backup_path}")
        sys.exit(0)

if __name__ == "__main__":
    main()