# Contributing to kandc

Thank you for your interest in contributing to kandc! This document provides guidelines and information for contributors.

## Development Workflow

### Setting up Development Environment

```bash
# Clone the repository
git clone https://github.com/Herdora/kandc.git
cd kandc

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e .
pip install pytest ruff mypy build twine
```

### Making Changes

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow the existing code style
   - Add tests for new functionality
   - Update documentation as needed

3. **Run tests locally**
   ```bash
   # Linting
   ruff check .
   ruff format .
   
   # Tests
   pytest tests/ -v
   
   # Type checking
   mypy src/kandc --ignore-missing-imports
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

5. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   # Create PR on GitHub
   ```

## Release Process

### Development Releases (kandc-dev)

Development releases are published automatically when you push to `dev` or `testing` branches:

```bash
git checkout dev
git merge your-feature-branch
git push origin dev
# Automatic kandc-dev release will be triggered
```

### Stable Releases (kandc)

Stable releases are triggered by git tags:

```bash
# 1. Ensure main branch is ready
git checkout main
git merge dev

# 2. Update version in pyproject.toml (if not done automatically)
# 3. Create and push tag
git tag v0.0.13
git push origin v0.0.13

# 4. The workflow will automatically:
#    - Run full test suite
#    - Build and publish to PyPI
#    - Create GitHub release
```

## Code Style

- **Linting**: We use `ruff` for linting and formatting
- **Type hints**: Add type hints for public APIs
- **Docstrings**: Use Google-style docstrings for functions and classes
- **Line length**: 100 characters maximum

Example:
```python
def capture_model_instance(
    model_instance: Any,
    model_name: Optional[str] = None,
    record_shapes: bool = True,
    profile_memory: bool = True,
    **kwargs: Any,
) -> Any:
    """Wrap a model instance to profile every forward pass.

    Args:
        model_instance: The model instance to wrap
        model_name: Name for the model traces (defaults to model class name)
        record_shapes: Record tensor shapes for each operation
        profile_memory: Profile memory usage
        **kwargs: Additional profiler arguments

    Returns:
        Wrapped model instance that profiles every forward pass
    """
```

## Testing

### Test Structure
```
tests/
├── test_auth.py          # Authentication tests
├── test_cli.py           # CLI functionality tests
├── test_timing.py        # Timing decorator tests
└── test_trace.py         # Trace capture tests
```

### Writing Tests
- Use `pytest` for all tests
- Mock external dependencies (API calls, file system)
- Test both success and failure cases
- Add integration tests for critical paths

### Running Tests
```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_trace.py -v

# With coverage
pytest tests/ --cov=kandc --cov-report=html
```

## CI/CD Pipeline

Our CI/CD pipeline runs on every push and PR:

### Automated Testing (`test.yml`)
- Tests across Python 3.8-3.12
- Linting and formatting checks
- Type checking with mypy
- Package installation verification
- Example code validation

### Publishing Workflows
- **Stable releases**: Triggered by version tags (`v*.*.*`)
- **Dev releases**: Triggered by pushes to dev/testing branches

## Package Structure

```
kandc/
├── src/kandc/           # Main package code
│   ├── __init__.py      # Public API exports
│   ├── auth.py          # Authentication handling
│   ├── cli.py           # Command-line interface
│   ├── constants.py     # Configuration constants
│   ├── timing.py        # Timing decorators
│   └── trace.py         # Model tracing functionality
├── tests/               # Test suite
├── examples/            # Usage examples
└── .github/workflows/   # CI/CD workflows
```

## Documentation

- **README.md**: Main package documentation
- **API docs**: Docstrings in code
- **Examples**: Comprehensive examples in `examples/` directory
- **Contributing**: This file

## Getting Help

- **Issues**: Open GitHub issues for bugs and feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Discord**: Join our community Discord (link in README)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
