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

## Tests

Run unit and integration tests relevant to changed modules.
For heavy Genesis-dependent scenarios, use explicit test gates where provided.

Reference:
- [tests](https://github.com/Royalvice/OmniNav/tree/main/tests)
