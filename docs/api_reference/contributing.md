# Contributing

## Development workflow

1. Read project contribution docs first:
- [REQUIREMENTS.md](https://github.com/Royalvice/OmniNav/blob/main/.github/contributing/REQUIREMENTS.md)
- [IMPLEMENTATION_PLAN.md](https://github.com/Royalvice/OmniNav/blob/main/.github/contributing/IMPLEMENTATION_PLAN.md)
- [TASK.md](https://github.com/Royalvice/OmniNav/blob/main/.github/contributing/TASK.md)
- [WALKTHROUGH.md](https://github.com/Royalvice/OmniNav/blob/main/.github/contributing/WALKTHROUGH.md)

2. Follow AGENTS constraints:
- [AGENTS.md](https://github.com/Royalvice/OmniNav/blob/main/AGENTS.md)

## Code expectations

- Batch-first data contracts `(B, ...)`
- Types centralized in `omninav/core/types.py`
- Registry-driven construction (no hardcoded instantiation)
- Lifecycle compliance for major components

## Docs expectations

When architecture/runtime/interface behavior changes, sync docs consistently:
- Requirements / Plan / Task / Walkthrough / Agents

Doc quality gates for this repository:
- Sphinx build must pass with `0 warning`.
- GitHub `blob/main` links in docs must resolve to real files.
- EN/ZH docs tree must keep mirrored page structure.
- Terminology and title style must pass docs style checks.

## Tests

Run unit and integration tests relevant to changed modules.
For heavy Genesis-dependent scenarios, use explicit test gates where provided.

Reference:
- [tests](https://github.com/Royalvice/OmniNav/tree/main/tests)

## Docs validation commands

```bash
python scripts/docs/check_repo_links.py
python scripts/docs/check_bilingual_structure.py
python scripts/docs/check_docs_style.py
source ~/omninav_ros_env/bin/activate && sphinx-build -b html docs docs/_build/html -W --keep-going
```
