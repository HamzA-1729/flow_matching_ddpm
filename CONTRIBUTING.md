# Contributing

Thanks for your interest! This project is a work in progress; issues and PRs are welcome.

## Setup

```bash
uv sync
uv run pre-commit install   # optional but recommended
```

## Before opening a PR

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

## Guidelines

- Keep DDPM and Flow Matching on the **same backbone, data and compute budget**. The comparison must stay fair.
- Add or update tests for any behaviour change.
- Use [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, `test:`, `chore:`).
- Type-hint public functions and add docstrings with tensor shapes.
