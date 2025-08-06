# Development

Guide for contributing to and developing Chisel CLI with the simplified API.

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
pip install -e .

# Install development dependencies
pip install ruff pytest
```

### Verify Installation

```bash
# Check CLI works
chisel --version

# Test import
python -c "from chisel import capture_trace; print('✅ Import successful')"

# Run examples
python examples/simple_example.py
```

## Project Structure

```
chisel/
├── src/chisel/               # Main package
│   ├── __init__.py          # Package entry point, exports capture_trace
│   ├── cli.py               # CLI implementation (main changes)
│   ├── trace.py             # capture_trace decorator
│   ├── auth.py              # Authentication service
│   ├── cached_files.py      # Large file caching
│   ├── constants.py         # GPU types and configuration
│   └── spinner.py           # UI utilities
├── examples/                # Working examples
│   ├── simple_example.py    # Basic usage
│   ├── args_example.py      # Command line arguments
│   ├── requirements_example.py # Custom requirements
│   └── specific_call.py     # Inline tracing
├── docs/                    # Documentation
├── tests/                   # Test suite
├── pyproject.toml          # Package configuration
└── README.md               # Main project README
```

## Code Style

### Ruff for Linting and Formatting

```bash
# Check code style
ruff check src/ examples/ tests/

# Auto-fix issues
ruff check src/ examples/ tests/ --fix

# Format code
ruff format src/ examples/ tests/

# Check formatting without changes
ruff format src/ examples/ tests/ --check
```

### Standards
- **Line length**: 100 characters
- **Quote style**: Double quotes
- **Indentation**: 4 spaces
- **Import order**: Standard library, third-party, local

### Pre-commit Hooks

```bash
pip install pre-commit

# Create .pre-commit-config.yaml
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

**CLI (`cli.py`):**
- Main entry point via `chisel` command
- Interactive configuration prompts
- Job submission and file upload
- Real-time output streaming
- Authentication management

**Trace Decorator (`trace.py`):**
- `capture_trace` decorator implementation
- GPU profiling and trace generation
- Environment detection (local vs cloud)
- Chrome trace format export

**Authentication (`auth.py`):**
- Browser-based OAuth flow
- Secure credential storage
- API key management
- Session validation

**Constants (`constants.py`):**
- Environment variable names
- `GPUType` enum for GPU configurations
- Default configuration values

**Cached Files (`cached_files.py`):**
- Large file detection and caching
- SHA256-based deduplication
- Transparent file replacement system

### Execution Flow

1. **Local Mode** (`python script.py`):
   - `capture_trace` decorator is pass-through
   - Functions run normally on local machine
   - No GPU profiling or job submission

2. **Cloud Mode** (`chisel python script.py`):
   - CLI handles authentication and configuration
   - Code packaged and uploaded to backend
   - `CHISEL_BACKEND_RUN=1` set on cloud execution
   - `capture_trace` decorator activates GPU profiling

3. **Backend Execution**:
   - Environment variables set by backend
   - `capture_trace` generates performance traces
   - Results and traces saved to shared volume

## Testing

### Run Tests

```bash
# All tests
pytest

# Verbose output
pytest -v

# Specific test file
pytest tests/test_cli.py

# With coverage
pytest --cov=chisel --cov-report=html
```

### Test Structure

```
tests/
├── test_cli.py          # CLI functionality
├── test_trace.py        # capture_trace decorator
├── test_auth.py         # Authentication
├── test_constants.py    # Constants and enums
└── conftest.py          # Pytest configuration
```

### Example Tests

```python
import pytest
import os
from unittest.mock import patch, MagicMock
from chisel import capture_trace
from chisel.constants import GPUType

def test_capture_trace_local():
    """Test capture_trace in local mode."""
    @capture_trace(trace_name="test")
    def test_function():
        return 42
    
    # Should be pass-through in local mode
    result = test_function()
    assert result == 42

def test_capture_trace_backend():
    """Test capture_trace in backend mode."""
    with patch.dict(os.environ, {
        'CHISEL_BACKEND_RUN': '1',
        'CHISEL_BACKEND_APP_NAME': 'test-app',
        'CHISEL_JOB_ID': 'job-123'
    }):
        @capture_trace(trace_name="test")
        def test_function():
            return 42
        
        # Should activate tracing in backend mode
        result = test_function()
        assert result == 42

def test_gpu_types():
    """Test GPU type enum."""
    assert GPUType.A100_80GB_1.value == "A100-80GB:1"
    assert GPUType.A100_80GB_4.value == "A100-80GB:4"
```

### Testing Guidelines

1. **Mock external dependencies** (network, filesystem, browser)
2. **Test environment isolation** with `patch.dict()`
3. **Test both local and cloud modes**
4. **Test error conditions and edge cases**
5. **Use fixtures for common setup**

## Contributing

### Workflow

1. **Fork repository** on GitHub
2. **Create feature branch**: `git checkout -b feature/description`
3. **Make changes**: Code + tests + docs
4. **Test changes**:
   ```bash
   ruff check src/ examples/ tests/
   ruff format src/ examples/ tests/ --check
   pytest
   ```
5. **Commit**: `git commit -m "Add feature: description"`
6. **Push**: `git push origin feature/description`
7. **Create Pull Request**

### Commit Messages

Use descriptive commit messages:

```bash
# Good
git commit -m "Add support for custom requirements files in CLI"
git commit -m "Fix argument parsing when mixing chisel and script flags"
git commit -m "Update documentation for simplified API"

# Poor
git commit -m "fix bug"
git commit -m "updates"
git commit -m "wip"
```

### Documentation Updates

When adding features, update relevant documentation:

- **API changes**: Update `docs/api-reference.md`
- **New examples**: Add to `docs/examples.md` and `examples/`
- **CLI changes**: Update `docs/configuration.md`
- **Breaking changes**: Update `README.md` and `docs/getting-started.md`

## Adding Features

### New CLI Options

1. **Update CLI parser** in `cli.py`:
```python
def parse_command_line_args(self, args: List[str]) -> Optional[Dict[str, Any]]:
    parser = argparse.ArgumentParser(...)
    
    # Add new option
    parser.add_argument(
        "--new-option",
        help="Description of new option",
        default="default_value"
    )
```

2. **Update interactive prompts**:
```python
def get_user_inputs_interactive(self, script_path: str = "<script.py>") -> Dict[str, Any]:
    # Add interactive prompt
    new_value = self.get_input_with_default("New option prompt", default="default")
    
    return {
        # ... existing options
        "new_option": new_value,
    }
```

3. **Add tests and documentation**

### New GPU Types

1. **Update enum** in `constants.py`:
```python
class GPUType(Enum):
    A100_80GB_1 = "A100-80GB:1"
    A100_80GB_2 = "A100-80GB:2"
    A100_80GB_4 = "A100-80GB:4"
    A100_80GB_8 = "A100-80GB:8"
    H100_80GB_1 = "H100-80GB:1"  # New type
```

2. **Update CLI options** in `cli.py`:
```python
self.gpu_options = [
    ("1", "A100-80GB:1", "Single GPU - Development, inference"),
    ("2", "A100-80GB:2", "2x GPUs - Medium training"),
    ("4", "A100-80GB:4", "4x GPUs - Large models"),
    ("8", "A100-80GB:8", "8x GPUs - Massive models"),
    ("h1", "H100-80GB:1", "Single H100 - High performance"),  # New option
]
```

3. **Update documentation** in `configuration.md`

### New Trace Options

1. **Update trace decorator** in `trace.py`:
```python
def capture_trace(
    trace_name: Optional[str] = None,
    record_shapes: bool = False,
    profile_memory: bool = False,
    new_option: bool = False,  # New option
    **profiler_kwargs: Any,
) -> Callable:
```

2. **Update profiler configuration**:
```python
with profile(
    activities=activities,
    record_shapes=record_shapes,
    profile_memory=profile_memory,
    new_feature=new_option,  # Use new option
    **profiler_kwargs
) as prof:
```

3. **Add tests and examples**

## Development Tips

### Local Development

Test the CLI without cloud submission:

```python
# Test capture_trace locally
from chisel import capture_trace

@capture_trace(trace_name="local_test")
def test_function():
    return "Hello, World!"

# Should be pass-through
result = test_function()
print(result)  # "Hello, World!"
```

### Debug CLI

```bash
# Enable debug output
export CHISEL_DEBUG=1
chisel python script.py

# Test CLI parsing
python -c "
from chisel.cli import ChiselCLI
cli = ChiselCLI()
config = cli.parse_command_line_args(['python', 'test.py', '--app-name', 'test'])
print(config)
"
```

### Mock Backend

For testing CLI functionality:

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/health')
def health():
    return {'status': 'ok'}

@app.route('/api/v1/submit-cachy-job-new', methods=['POST'])
def submit_job():
    return {
        'job_id': 'test-123',
        'exit_code': 0,
        'message': 'Job submitted successfully',
        'visit_url': '/jobs/test-123'
    }

if __name__ == '__main__':
    app.run(port=8000, debug=True)
```

```bash
# Run mock backend
python mock_backend.py

# Test CLI against mock
CHISEL_BACKEND_URL=http://localhost:8000 chisel python script.py
```

### Performance Profiling

```python
import cProfile
import pstats
from chisel.cli import ChiselCLI

def profile_cli():
    cli = ChiselCLI()
    cli.parse_command_line_args(['python', 'test.py', '--app-name', 'test'])

# Profile CLI performance
cProfile.run('profile_cli()', 'cli_profile.stats')
stats = pstats.Stats('cli_profile.stats')
stats.sort_stats('cumulative').print_stats(10)
```

## Release Process

### Version Management

Update version in:
- `pyproject.toml`: `version = "0.2.0"`
- `src/chisel/__init__.py`: `__version__ = "0.2.0"`

### Release Checklist

1. **Update version numbers**
2. **Run full test suite**: `pytest`
3. **Test examples manually**:
   ```bash
   python examples/simple_example.py
   python examples/args_example.py --iterations 3
   ```
4. **Update documentation**
5. **Create git tag**: `git tag v0.2.0`
6. **Push tag**: `git push origin v0.2.0`

### Testing Release

```bash
# Build package
python -m build

# Test installation
pip install dist/chisel_cli-0.2.0-py3-none-any.whl

# Verify CLI works
chisel --version
chisel python examples/simple_example.py
```

## IDE Setup

### VS Code

`.vscode/settings.json`:
```json
{
    "python.defaultInterpreterPath": "./.venv/bin/python",
    "[python]": {
        "editor.formatOnSave": true,
        "editor.codeActionsOnSave": {
            "source.fixAll.ruff": true
        }
    },
    "python.testing.pytestEnabled": true,
    "python.testing.unittestEnabled": false
}
```

### PyCharm

1. Set interpreter to `.venv/bin/python`
2. Install Ruff plugin
3. Configure Ruff as formatter and linter
4. Enable pytest as test runner

## Community

### Getting Help

- **📧 Email**: [contact@herdora.com](mailto:contact@herdora.com) - Direct support
- **💬 GitHub Discussions**: Questions and ideas
- **🐛 GitHub Issues**: Bug reports and feature requests
- **📖 Documentation**: Latest docs and guides

### Guidelines

- Be respectful and follow code of conduct
- Start with small improvements or bug fixes
- Ask questions when unclear about implementation
- Test thoroughly before submitting
- Update documentation for user-facing changes

### Code Review Process

1. **Automated checks**: Ruff, pytest, type checking
2. **Manual review**: Code quality, design, documentation
3. **Testing**: Verify examples work, test edge cases
4. **Documentation**: Ensure docs are updated appropriately

Thank you for contributing to Chisel CLI! 🚀

**Next:** [API Reference](api-reference.md) | [Examples](examples.md)