# CI/CD Workflows for kandc

This directory contains GitHub Actions workflows for automated testing and publishing of the kandc package.

## Workflows

### 1. `test.yml` - Continuous Testing
**Triggers:** Push to main/dev/testing branches, Pull requests
**Purpose:** Run comprehensive tests, linting, and quality checks

- Tests across Python 3.8-3.12
- Linting with ruff
- Type checking with mypy
- Package installation tests
- Example code validation
- Build verification for both kandc and kandc-dev

### 2. `publish-stable.yml` - Stable Release Publishing
**Triggers:** Git tags matching `v*.*.*` (e.g., v0.0.13, v1.0.0)
**Purpose:** Publish stable releases to PyPI as `kandc`

- Runs full test suite before publishing
- Automatically extracts version from git tag
- Updates pyproject.toml version
- Publishes to PyPI
- Creates GitHub release with artifacts

### 3. `publish-dev.yml` - Development Release Publishing
**Triggers:** Push to dev/testing branches, Manual workflow dispatch
**Purpose:** Publish development releases to PyPI as `kandc-dev`

- Runs tests before publishing
- Generates unique dev version numbers
- Publishes to PyPI as kandc-dev
- Comments on commits with install instructions

## Required GitHub Secrets

You need to set up the following secrets in your GitHub repository:

### For Stable Releases (`kandc`)
1. Go to your repo → Settings → Secrets and variables → Actions
2. Add secret: `PYPI_API_TOKEN`
   - Value: Your PyPI API token for the `kandc` package
   - Get from: https://pypi.org/manage/account/token/

### For Dev Releases (`kandc-dev`)
3. Add secret: `PYPI_DEV_API_TOKEN`
   - Value: Your PyPI API token for the `kandc-dev` package
   - Get from: https://pypi.org/manage/account/token/

## GitHub Environments (Recommended)

Set up environments for additional protection:

1. Go to Settings → Environments
2. Create environment: `pypi-stable`
   - Add protection rules (require reviews, restrict branches)
   - Add secret: `PYPI_API_TOKEN`
3. Create environment: `pypi-dev`
   - Add secret: `PYPI_DEV_API_TOKEN`

## Usage

### Publishing Stable Release
```bash
# 1. Update version in pyproject.toml (optional - workflow will do this)
# 2. Create and push a git tag
git tag v0.0.13
git push origin v0.0.13

# The workflow will automatically:
# - Run tests
# - Build package
# - Publish to PyPI
# - Create GitHub release
```

### Publishing Dev Release
```bash
# Automatic on push to dev/testing branches
git checkout dev
git push origin dev

# Manual trigger with custom version
# Go to Actions → Publish kandc-dev → Run workflow
# Specify version suffix (e.g., "rc1", "dev5")
```

### Testing Changes
```bash
# Any push to main/dev/testing or PR will trigger tests
git push origin main
```

## Local Testing

Before pushing, you can run similar checks locally:

```bash
# Install dev dependencies
pip install pytest ruff mypy build twine

# Run linting
ruff check .
ruff format --check .

# Run tests
pytest tests/ -v

# Test build
python -m build
twine check dist/*
```

## Troubleshooting

### Common Issues

1. **PyPI upload fails with "File already exists"**
   - Version numbers must be unique
   - Dev workflow generates unique timestamps
   - Stable workflow uses git tag version

2. **Tests fail in CI but pass locally**
   - Check Python version compatibility
   - Ensure all dependencies are specified
   - Check for missing test files in git

3. **Permission denied on PyPI**
   - Verify API tokens are correct
   - Check token permissions
   - Ensure package names match (kandc vs kandc-dev)

### Monitoring

- Check Actions tab for workflow runs
- Monitor PyPI for successful uploads
- Review GitHub releases for stable versions
- Check commit comments for dev version notifications
