#!/bin/bash
# Setup script for kandc CI/CD

set -e

echo "🚀 Setting up CI/CD for kandc"
echo "================================"

# Check if we're in the right directory
if [[ ! -f "pyproject.toml" ]] || [[ ! -f "pyproject.dev.toml" ]]; then
    echo "❌ Error: Please run this script from the kandc root directory"
    exit 1
fi

# Check if GitHub CLI is installed
if ! command -v gh &> /dev/null; then
    echo "❌ GitHub CLI (gh) is not installed."
    echo "   Install it from: https://cli.github.com/"
    echo "   Or use the GitHub web interface to set up secrets manually."
    exit 1
fi

# Check if user is logged in to GitHub CLI
if ! gh auth status &> /dev/null; then
    echo "🔐 Please log in to GitHub CLI first:"
    echo "   gh auth login"
    exit 1
fi

echo "✅ GitHub CLI is ready"

# Get repository info
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
echo "📁 Repository: $REPO"

echo ""
echo "🔑 Setting up GitHub Secrets"
echo "=============================="

# Function to set secret
set_secret() {
    local secret_name=$1
    local secret_description=$2
    
    echo ""
    echo "Setting up: $secret_name"
    echo "Description: $secret_description"
    echo ""
    
    if gh secret list | grep -q "^$secret_name"; then
        echo "⚠️  Secret $secret_name already exists."
        read -p "Do you want to update it? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "⏭️  Skipping $secret_name"
            return
        fi
    fi
    
    echo "Please enter your $secret_name:"
    echo "(Get it from: https://pypi.org/manage/account/token/)"
    read -s -p "Token: " token
    echo
    
    if [[ -z "$token" ]]; then
        echo "⏭️  Empty token, skipping $secret_name"
        return
    fi
    
    gh secret set "$secret_name" --body "$token"
    echo "✅ Secret $secret_name has been set"
}

# Set up secrets
set_secret "PYPI_API_TOKEN" "PyPI API token for publishing stable 'kandc' releases"
set_secret "PYPI_DEV_API_TOKEN" "PyPI API token for publishing 'kandc-dev' releases"

echo ""
echo "🌍 Setting up GitHub Environments (Optional but Recommended)"
echo "==========================================================="
echo ""
echo "For additional security, you can set up GitHub Environments:"
echo "1. Go to: https://github.com/$REPO/settings/environments"
echo "2. Create environment: 'pypi-stable'"
echo "   - Add protection rules (require reviews, restrict to main branch)"
echo "   - Add secret: PYPI_API_TOKEN"
echo "3. Create environment: 'pypi-dev'"
echo "   - Add secret: PYPI_DEV_API_TOKEN"
echo ""

echo "🧪 Testing CI/CD Setup"
echo "======================"
echo ""
echo "To test your CI/CD setup:"
echo ""
echo "1. Test the test workflow:"
echo "   git push origin main"
echo ""
echo "2. Test dev publishing (if on dev branch):"
echo "   git checkout dev"
echo "   git push origin dev"
echo ""
echo "3. Test stable publishing:"
echo "   git tag v0.0.13"
echo "   git push origin v0.0.13"
echo ""

echo "✅ CI/CD setup complete!"
echo ""
echo "📚 Next steps:"
echo "   - Review .github/workflows/README.md for detailed information"
echo "   - Check .github/CONTRIBUTING.md for development guidelines"
echo "   - Monitor the Actions tab for workflow runs"
echo ""
echo "🎉 Happy coding!"
