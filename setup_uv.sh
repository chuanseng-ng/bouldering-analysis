#!/bin/bash
# Setup script for bouldering-analysis using uv
# This script installs uv, creates a virtual environment, and installs dependencies

set -e  # Exit on error

echo "🚀 Setting up bouldering-analysis with uv..."
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "📦 uv not found. Installing uv..."

    # Detect OS and install accordingly
    if [[ "$OSTYPE" == "linux-gnu"* ]] || [[ "$OSTYPE" == "darwin"* ]]; then
        if curl -LsSf https://astral.sh/uv/install.sh | sh; then
            # Add uv to PATH for this session
            export PATH="$HOME/.local/bin:$PATH"
            echo "✅ uv installed successfully"
        else
            echo "❌ Failed to install uv" >&2
            exit 1
        fi
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        echo "⚠️  Please install uv manually on Windows:"
        echo "    PowerShell: powershell -ExecutionPolicy ByPass -c \"irm https://astral.sh/uv/install.ps1 | iex\""
        echo "    Or using pip: pip install uv"
        exit 1
    else
        echo "❌ Unsupported or unknown operating system: $OSTYPE" >&2
        echo "Please install uv manually from: https://github.com/astral-sh/uv" >&2
        exit 1
    fi
else
    echo "✅ uv is already installed ($(uv --version))"
fi

echo ""
echo "🐍 Setting up Python environment..."

# Install Python 3.11 if not available
if ! uv python list | grep -Eq '3\.11($|\.)'; then
    echo "📥 Installing Python 3.11..."
    if ! uv python install 3.11; then
        echo "❌ Failed to install Python 3.11" >&2
        exit 1
    fi
fi

echo ""
echo "📦 Creating virtual environment and installing dependencies..."

# Sync dependencies (creates venv if needed)
# Note: --no-install-project is used because this is an application, not a library
# --all-extras installs dev dependencies (testing, linting, etc.)
if ! uv sync --no-install-project --all-extras; then
    echo "❌ Failed to sync dependencies" >&2
    exit 1
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "📝 Next steps:"
echo "   1. Activate the virtual environment:"
echo "      - macOS/Linux: source .venv/bin/activate"
printf '%s\n' '      - Windows (PowerShell): .venv\Scripts\Activate.ps1'
printf '%s\n' '      - Windows (CMD): .venv\Scripts\activate.bat'
echo ""
echo "   2. Or run commands directly with uv:"
echo "      - uv run pytest tests/"
echo "      - uv run uvicorn src.app:application --reload"
echo ""
echo "   3. Configure Supabase (required for upload functionality):"
echo "      - See docs/SUPABASE_SETUP.md"
echo ""
echo "📚 For more information, see docs/UV_SETUP.md"
