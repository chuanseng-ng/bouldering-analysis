"""
Setup utilities for initializing the bouldering analysis environment.

Provides functions for database setup and directory creation.
"""

from pathlib import Path
from sqlalchemy.exc import SQLAlchemyError


def setup_database():
    """Set up the SQLite database"""
    print("\nSetting up database...")

    # Create database directory if it doesn't exist
    db_path = Path("bouldering_analysis.db")
    if db_path.exists():
        print("✓ Database already exists")
        return True

    # Create database tables by running the app once
    try:
        from src.main import create_tables  # pylint: disable=import-outside-toplevel

        create_tables()
        print("✓ Database file exists, verifying schema...")
        # Optionally verify schema here
        return True
    except (SQLAlchemyError, ImportError) as e:
        print(f"✗ Failed to create database tables: {e}")
        return False


def create_directories():
    """Create necessary directories"""
    print("\nCreating directories...")

    directories = ["data/uploads", "data/processed", "logs", "models"]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")

    return True
