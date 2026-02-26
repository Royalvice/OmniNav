#!/usr/bin/env python3
"""Validate docs GitHub blob links and local markdown links."""

from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
DOC_GLOBS = ["docs/**/*.md", "README.md", "README_CN.md"]

BLOB_PATTERN = re.compile(
    r"\[[^\]]+\]\(https://github\.com/Royalvice/OmniNav/blob/main/([^\)\s#]+)\)"
)
LOCAL_MD_PATTERN = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")


def iter_target_files() -> list[pathlib.Path]:
    files: list[pathlib.Path] = []
    for pattern in DOC_GLOBS:
        files.extend(ROOT.glob(pattern))
    return sorted(set(files))


def check_blob_links(path: pathlib.Path, text: str, errors: list[str]) -> None:
    for match in BLOB_PATTERN.finditer(text):
        rel = match.group(1)
        target = ROOT / rel
        if not target.exists():
            errors.append(f"{path}: missing blob target -> {rel}")


def check_local_links(path: pathlib.Path, text: str, errors: list[str]) -> None:
    for match in LOCAL_MD_PATTERN.finditer(text):
        url = match.group(1).strip()
        if not url or url.startswith(("http://", "https://", "mailto:", "#")):
            continue
        if url.startswith("../") or url.startswith("./") or url.startswith("/") or url.endswith(".md"):
            target = (path.parent / url).resolve() if not url.startswith("/") else (ROOT / url.lstrip("/"))
            # Sphinx doc links can omit suffix, skip those without extension.
            if pathlib.Path(url).suffix in {"", ".html"}:
                continue
            if not target.exists():
                errors.append(f"{path}: broken local link -> {url}")


def main() -> int:
    errors: list[str] = []
    for md in iter_target_files():
        text = md.read_text(encoding="utf-8")
        check_blob_links(md, text, errors)
        check_local_links(md, text, errors)

    if errors:
        print("[check_repo_links] FAILED")
        for err in errors:
            print(f"- {err}")
        return 1

    print("[check_repo_links] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
