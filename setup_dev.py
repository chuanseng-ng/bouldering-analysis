#!/usr/bin/env python3
"""
Development environment setup script for Bouldering Route Analysis
"""

import sys
import subprocess
from pathlib import Path
from src.setup import setup_database, create_directories


def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{description}...")
    try:
        result = subprocess.run(
            command, shell=True, check=True, capture_output=True, text=True
        )
        print(f"✓ {description} completed successfully")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed")
        print(f"Error: {e.stderr}")
        return False


def verify_installation():
    """Verify the installation"""
    print("\nVerifying installation...")

    try:
        # Test imports
        from src.main import app  # noqa: F401
        from src.models import db, Analysis, Feedback  # noqa: F401
        from PIL import Image  # noqa: F401
        from ultralytics import YOLO  # noqa: F401
        from sqlalchemy import text

        print("✓ All imports successful")

        # Test database connection
        with app.app_context():
            db.session.execute(text("SELECT 1"))
        print("✓ Database connection successful")

        # Test if YOLO model can be loaded
        try:
            _ = YOLO("yolov8n.pt")  # Assign to _ to indicate intentional discard
            print("✓ YOLO model loaded successfully")
        except Exception as e:
            print(f"⚠ YOLO model loading failed: {e}")
            print("  This might be normal if the model file is not available")

        return True

    except Exception as e:
        print(f"✗ Verification failed: {e}")
        return False


def main():
    """Main setup function"""
    print("Bouldering Route Analysis - Development Setup")
    print("=" * 50)

    # Check if we're in the right directory
    if not Path("src").exists() or not Path("requirements.txt").exists():
        print("✗ Please run this script from the project root directory")
        return False

    # Run setup steps
    steps = [
        (create_directories, "Creating directories"),
        (setup_database, "Setting up database"),
        (verify_installation, "Verifying installation"),
    ]

    for step_func, step_name in steps:
        if not step_func():
            print(f"\n✗ Setup failed at: {step_name}")
            return False

    print("\n" + "=" * 50)
    print("✓ Development environment setup completed successfully!")
    print("\nNext steps:")
    print("1. Run 'python src/main.py' to start the development server")
    print("2. Open http://localhost:5000 in your browser")
    print("3. Upload a bouldering route image to test the analysis")

    return True


if __name__ == "__main__":
    SUCCESS = main()
    sys.exit(0 if SUCCESS else 1)
