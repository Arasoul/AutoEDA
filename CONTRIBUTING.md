# Contributing to AutoEDA

Thank you for considering contributing to AutoEDA!

## Development Setup

```bash
git clone https://github.com/Arasoul/AutoEDA.git
cd AutoEDA
pip install -e ".[dev]"
```

## Code Style

- PEP 8 compliant
- Type hints on every public function
- Google-style docstrings
- Use `logging` instead of `print()`
- `pathlib` instead of `os.path`

## Running Checks

```bash
ruff check src/ tests/
mypy src/
pytest tests/ -v --cov=autoeda --cov-report=term-missing
```

## Pull Requests

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Ensure all checks pass
5. Submit a pull request

## Testing

Every new feature must include corresponding unit tests.
Target at least 90% code coverage.
