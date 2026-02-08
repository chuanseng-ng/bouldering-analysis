# Setup script for bouldering-analysis using uv (PowerShell)
# This script installs uv, creates a virtual environment, and installs dependencies

$ErrorActionPreference = "Stop"

Write-Host "🚀 Setting up bouldering-analysis with uv..." -ForegroundColor Cyan
Write-Host ""

# Check if uv is installed
$uvInstalled = Get-Command uv -ErrorAction SilentlyContinue

if (-not $uvInstalled) {
    Write-Host "📦 uv not found. Installing uv..." -ForegroundColor Yellow

    # Install uv using the official installer
    try {
        Invoke-RestMethod -Uri "https://astral.sh/uv/install.ps1" | Invoke-Expression

        # Refresh PATH for this session
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

        Write-Host "✅ uv installed successfully" -ForegroundColor Green
    }
    catch {
        Write-Host "❌ Failed to install uv automatically" -ForegroundColor Red
        Write-Host "Please install manually:" -ForegroundColor Yellow
        Write-Host "  Option 1: pip install uv" -ForegroundColor White
        Write-Host "  Option 2: Download from https://github.com/astral-sh/uv/releases" -ForegroundColor White
        exit 1
    }
}
else {
    $uvVersion = & uv --version
    Write-Host "✅ uv is already installed ($uvVersion)" -ForegroundColor Green
}

Write-Host ""
Write-Host "🐍 Setting up Python environment..." -ForegroundColor Cyan

# Install Python 3.11 if not available
$pythonList = & uv python list
if ($pythonList -notmatch "3.11") {
    Write-Host "📥 Installing Python 3.11..." -ForegroundColor Yellow
    & uv python install 3.11
}

Write-Host ""
Write-Host "📦 Creating virtual environment and installing dependencies..." -ForegroundColor Cyan

# Sync dependencies (creates venv if needed)
# Note: --no-install-project is used because this is an application, not a library
# --all-extras installs dev dependencies (testing, linting, etc.)
& uv sync --no-install-project --all-extras

Write-Host ""
Write-Host "✅ Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "📝 Next steps:" -ForegroundColor Cyan
Write-Host "   1. Activate the virtual environment:" -ForegroundColor White
Write-Host "      .venv\Scripts\Activate.ps1" -ForegroundColor Yellow
Write-Host ""
Write-Host "   2. Or run commands directly with uv:" -ForegroundColor White
Write-Host "      uv run pytest tests/" -ForegroundColor Yellow
Write-Host "      uv run uvicorn src.app:application --reload" -ForegroundColor Yellow
Write-Host ""
Write-Host "   3. Configure Supabase (required for upload functionality):" -ForegroundColor White
Write-Host "      See docs/SUPABASE_SETUP.md" -ForegroundColor Yellow
Write-Host ""
Write-Host "📚 For more information, see docs/UV_SETUP.md" -ForegroundColor Cyan
