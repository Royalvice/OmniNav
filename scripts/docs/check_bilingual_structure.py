#!/usr/bin/env python3
"""Ensure EN docs pages have ZH mirrors and vice versa (with allowed exceptions)."""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
ZH = DOCS / "zh"

# zh/index.md intentionally orphan from EN toctree, but still required as mirror of docs/index.md.
EXTRA_EN_ONLY = {
    "requirements.txt",
}


def md_files(base: pathlib.Path) -> set[str]:
    out: set[str] = set()
    for p in base.rglob("*.md"):
        out.add(str(p.relative_to(base)))
    return out


def main() -> int:
    errors: list[str] = []

    en = md_files(DOCS)
    zh = md_files(ZH)

    en_without_zh_prefix = {p for p in en if not p.startswith("zh/")}

    en_expected_mirror = {
        p for p in en_without_zh_prefix if p not in EXTRA_EN_ONLY
    }
    zh_expected = {f"zh/{p}" for p in en_expected_mirror}

    zh_actual = {f"zh/{p}" for p in zh}

    missing_zh = sorted(zh_expected - zh_actual)
    extra_zh = sorted(zh_actual - zh_expected)

    if missing_zh:
        errors.append("Missing ZH mirrors:")
        errors.extend(f"  - {p}" for p in missing_zh)
    if extra_zh:
        errors.append("Unexpected ZH-only files (no EN mirror):")
        errors.extend(f"  - {p}" for p in extra_zh)

    if errors:
        print("[check_bilingual_structure] FAILED")
        for line in errors:
            print(line)
        return 1

    print("[check_bilingual_structure] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
