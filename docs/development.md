# Development

Guide for contributing to and developing Chisel CLI.

## Development Setup

### Prerequisites
- Python 3.8+
- Git
- Virtual environment

### Installation

```bash
git clone https://github.com/Herdora/chisel.git
cd chisel

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# OR .venv\Scripts\activate  # Windows

# Install in development mode
pip install -e .[dev]
```

### Development Dependencies

```bash
pip install ruff pytest
```

## Project Structure

```
chisel/
├── src/chisel/               # Main package
│   ├── __init__.py          # CLI entry point
│   ├── core.py              # ChiselApp functionality
│   ├── auth.py              # Authentication
│   ├── spinner.py           # UI utilities
│   └── constants.py         # GPU types and config
├── examples/                # Usage examples
├── docs/                    # Documentation
├── pyproject.toml          # Package configuration
└── README.md               # Main project README
```

## Code Style

### Ruff for Linting and Formatting

```bash
# Check code style
ruff check src/ examples/

# Auto-fix issues
ruff check src/ examples/ --fix

# Format code
ruff format src/ examples/

# Check formatting
ruff format src/ examples/ --check
```

### Standards
- **Line length**: 100 characters
- **Quote style**: Double quotes
- **Indentation**: 4 spaces
- **Import order**: Standard library, third-party, local

### Pre-commit Hooks

```bash
pip install pre-commit

# .pre-commit-config.yaml
cat > .pre-commit-config.yaml << 'EOF'
repos:
- repo: https://github.com/astral-sh/ruff-pre-commit
  rev: v0.1.9
  hooks:
    - id: ruff
      args: [--fix]
    - id: ruff-format
EOF

pre-commit install
```

## Architecture

### Core Components

**ChiselApp (`core.py`):**
- GPU configuration and validation
- Code packaging and upload
- Job submission and tracking
- Function decoration and tracing

**Authentication (`auth.py`):**
- Browser-based authentication flow
- Secure credential storage
- API key management

**CLI Entry Point (`__init__.py`):**
- `chisel` command implementation
- Environment variable management
- Command execution and argument passing

**Constants (`constants.py`):**
- Environment variable names
- `GPUType` enum for GPU configurations
- Default values

### Execution Flow

1. **Local Mode** (`CHISEL_ACTIVATED != "1"`):
   - ChiselApp inactive mode
   - Decorators are pass-through
   - Runs on local machine

2. **Chisel Mode** (`chisel` command):
   - Sets `CHISEL_ACTIVATED=1`
   - Activates GPU functionality
   - Uploads and runs on cloud GPU

3. **Backend Mode** (`CHISEL_BACKEND_RUN=1`):
   - Detects backend execution
   - Skips auth and upload
   - Executes on cloud GPU

## Testing

### Run Tests

```bash
# All tests
pytest

# Verbose output
pytest -v

# Specific test file
pytest tests/test_core.py

# With coverage
pytest --cov=chisel
```

### Test Structure

```
tests/
├── test_core.py      # ChiselApp functionality
├── test_auth.py      # Authentication
├── test_cli.py       # CLI functionality
└── conftest.py       # Pytest configuration
```

### Example Test

```python
import pytest
import os
from unittest.mock import patch
from chisel import ChiselApp, GPUType

def test_gpu_type_conversion():
    app = ChiselApp("test", gpu=GPUType.A100_80GB_2)
    assert app.gpu == "A100-80GB:2"

def test_inactive_mode():
    with patch.dict(os.environ, {'CHISEL_ACTIVATED': ''}, clear=True):
        app = ChiselApp("test")
        assert not app.activated
```

### Testing Guidelines

1. Mock external dependencies
2. Test environment isolation with `patch.dict()`
3. Test both local and Chisel modes
4. Test error conditions and edge cases

## Contributing

### Workflow

1. **Fork repository** on GitHub
2. **Create feature branch**: `git checkout -b feature/name`
3. **Make changes**: Code + tests + docs
4. **Test changes**:
   ```bash
   ruff check src/ examples/
   ruff format src/ examples/ --check
   pytest
   ```
5. **Commit**: `git commit -m "Add feature: description"`
6. **Push**: `git push origin feature/name`
7. **Create Pull Request**

### Commit Messages

```bash
# Good
git commit -m "Add support for custom GPU types"
git commit -m "Fix argument passing in CLI wrapper"

# Poor
git commit -m "fix bug"
git commit -m "updates"
```

### Documentation Updates

When adding features:
- **API changes**: Update `docs/api-reference.md`
- **New examples**: Add to `docs/examples.md`
- **Configuration**: Update `docs/configuration.md`
- **Breaking changes**: Update `README.md`

## Adding Features

### New GPU Types

1. **Update enum** in `constants.py`:
   ```python
   class GPUType(Enum):
       A100_80GB_1 = "A100-80GB:1"
       H100_80GB_1 = "H100-80GB:1"  # New type
   ```

2. **Update documentation** in `configuration.md`
3. **Add tests**

### New CLI Commands

1. **Extend CLI** in `__init__.py`:
   ```python
   def main():
       if len(sys.argv) < 2:
           print_help()
           return 0
       
       command = sys.argv[1]
       
       if command == "version":
           print(f"Chisel CLI v{__version__}")
           return 0
       # Add new commands here
   ```

2. **Add tests** and documentation

## Release Process

### Version Management

Update version in both:
- `pyproject.toml`: `version = "0.1.0"`
- `src/chisel/__init__.py`: `__version__ = "0.1.0"`

### Release Checklist

1. Update version numbers
2. Run full test suite
3. Test examples manually
4. Update documentation
5. Create git tag: `git tag v0.1.0`

## Development Tips

### Local Development

```python
import os

# Force inactive mode
os.environ.pop('CHISEL_ACTIVATED', None)

from chisel import ChiselApp
app = ChiselApp("dev-app")  # Inactive

@app.capture_trace()  # No-op
def test_function():
    return 42
```

### Debug Authentication

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from chisel.auth import AuthService
auth = AuthService()  # Debug info printed
```

### Mock Backend

```python
from flask import Flask

app = Flask(__name__)

@app.route('/health')
def health():
    return {'status': 'ok'}

@app.route('/api/v1/submit-cachy-job-new', methods=['POST'])
def submit_job():
    return {'job_id': 'test-123', 'exit_code': 0}

app.run(port=8000)
```

### Performance Profiling

```python
import cProfile
import pstats

def profile_function():
    # Your code here
    pass

cProfile.run('profile_function()', 'profile_stats')
stats = pstats.Stats('profile_stats')
stats.sort_stats('cumulative').print_stats(10)
```

## IDE Setup

### VS Code

`.vscode/settings.json`:
```json
{
    "python.defaultInterpreterPath": "./.venv/bin/python",
    "python.linting.ruffEnabled": true,
    "[python]": {
        "editor.formatOnSave": true,
        "editor.codeActionsOnSave": {
            "source.fixAll.ruff": true
        }
    }
}
```

### PyCharm

1. Set interpreter to `.venv/bin/python`
2. Install Ruff plugin
3. Configure Ruff as formatter and linter

## Community

### Getting Help
- **📧 Email**: [contact@herdora.com](mailto:contact@herdora.com) - Direct support
- **💬 GitHub Discussions**: Questions and ideas
- **🐛 GitHub Issues**: Bug reports and feature requests
- **📖 Documentation**: Latest docs and guides

### Guidelines
- Be respectful and follow code of conduct
- Start with small improvements or bug fixes
- Ask questions when unclear
- Test thoroughly before submitting

Thank you for contributing to Chisel CLI! 🚀

**Next:** [API Reference](api-reference.md) | [Examples](examples.md)