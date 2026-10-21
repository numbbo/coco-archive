# Development

## Running the Tests

The project uses [uv](https://docs.astral.sh/uv/). 
Install the package with all extras and the development dependencies (pytest, pytest-cov, ruff):

```
uv sync --all-extras --dev
```

Run the tests with coverage:

```
uv run pytest --cov=coco_archive
```

When you change an example, make it runnable on its own.
Define everything it uses in the same docstring and give the expected output of every expression.
Use `# doctest: +SKIP` only for examples that can't run, e.g. with a placeholder URL.

CI also lints the code with ruff, so check it before pushing:

```
uv run ruff check .
```

## Commit Convention

We follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification for all commits.

### Format

```
<type>: <description>

[optional body]

[optional footer(s)]
```

### Types

- `feat` -- A new feature
- `fix` -- A bug fix
- `chore` -- Maintenance tasks or dependency updates.
- `docs` -- Documentation changes
- `meta` -- Meta Changes
