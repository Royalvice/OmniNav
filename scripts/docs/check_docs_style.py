#!/usr/bin/env python3
"""Lightweight style checks for docs consistency."""

from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"

CJK_RE = re.compile(r"[\u4e00-\u9fff]")
H1_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)

FORBIDDEN_TERMS = {
    "WayPoint": "Waypoint",
    "ObjectNAV": "ObjectNav",
    "PointNAV": "PointNav",
    "nav22": "Nav2",
}


def read(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")


def main() -> int:
    errors: list[str] = []

    # Enforce no zh/index entry in EN toctree to keep top nav clean.
    en_index = read(DOCS / "index.md")
    if "\nzh/index\n" in en_index:
        errors.append("docs/index.md: must not include zh/index in toctree")

    for md in sorted(DOCS.rglob("*.md")):
        text = read(md)
        m = H1_RE.search(text)
        if not m:
            errors.append(f"{md}: missing H1 heading")
            continue
        h1 = m.group(1).strip()

        rel = md.relative_to(DOCS)
        is_zh = rel.parts[0] == "zh"
        has_cjk = bool(CJK_RE.search(h1))

        if is_zh and not has_cjk:
            errors.append(f"{md}: zh page H1 should include Chinese text -> '{h1}'")
        if (not is_zh) and has_cjk:
            errors.append(f"{md}: en page H1 should not include Chinese text -> '{h1}'")

        for bad, expected in FORBIDDEN_TERMS.items():
            if bad in text:
                errors.append(f"{md}: forbidden term '{bad}', use '{expected}'")

    if errors:
        print("[check_docs_style] FAILED")
        for err in errors:
            print(f"- {err}")
        return 1

    print("[check_docs_style] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
