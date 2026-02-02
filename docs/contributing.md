# Contributing Guide

Thank you for your interest in OmniNav! We welcome contributions of all forms.

## How to Contribute

### Reporting Bugs

1. Search [GitHub Issues](https://github.com/Royalvice/OmniNav/issues) to see if the issue already exists.
2. If not, create a new Issue including:
   - Clear problem description
   - Steps to reproduce
   - Expected vs. actual behavior
   - Environment info (OS, Python version, GPU, etc.)

### Submitting Code

1. Fork this repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -m "feat: add your feature"`
4. Push branch: `git push origin feature/your-feature`
5. Create a Pull Request

### Code Standards

- Use [Black](https://github.com/psf/black) to format code
- Use [isort](https://pycqa.github.io/isort/) to sort imports
- Follow [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- Add unit tests for new features

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/OmniNav.git
cd OmniNav

# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/
```

## Contributing to Documentation

Documentation is written using MkDocs + Material theme and is located in the `docs/` directory.

```bash
# Install doc dependencies
pip install mkdocs-material mkdocstrings[python]

# Local preview
mkdocs serve
```

## License

By contributing code, you agree that your contributions will be licensed under the project's Apache-2.0 license.
