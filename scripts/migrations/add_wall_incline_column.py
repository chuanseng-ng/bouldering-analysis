#!/usr/bin/env python3
"""
Migration script to add the wall_incline column to the analyses table.

WHEN TO RUN:
- Run this migration when upgrading to the Phase 1a grade prediction system
- This adds support for storing wall angle information for each route analysis

PREREQUISITES:
- Ensure you have a database backup before running this migration
- This migration is safe and adds a column with a default value

HOW TO VERIFY SUCCESS:
- Query the analyses table schema to confirm wall_incline column exists
- Verify existing records have 'vertical' as the default value
- Check that the application functions normally after migration

HOW TO ROLLBACK:
- Run this script with the --rollback flag to remove the column
"""

from __future__ import annotations

import os
import sys
import logging
import argparse
from pathlib import Path

from sqlalchemy import (
    create_engine,
    inspect,
    text,
)
from sqlalchemy.exc import SQLAlchemyError

# Add the project root to the Python path to allow imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Ensure logs directory exists before configuring FileHandler
project_root.joinpath("logs").mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(project_root / "logs" / "migration_add_wall_incline.log"),
    ],
)
logger = logging.getLogger(__name__)


def get_database_url() -> str:
    """
    Get the database URL from environment or use default.

    Uses the same configuration pattern as src/main.py.

    Returns:
        str: Database connection URL
    """
    db_url = os.environ.get("DATABASE_URL") or "sqlite:///bouldering_analysis.db"

    # Convert relative SQLite paths to absolute paths from project root
    if db_url.startswith("sqlite:///") and not db_url.startswith("sqlite:////"):
        db_path = db_url.replace("sqlite:///", "")
        if not os.path.isabs(db_path):
            db_path = str(project_root / db_path)
            db_url = f"sqlite:///{db_path}"

    return db_url


def get_database_type(db_url: str) -> str:
    """
    Determine the database type from the connection URL.

    Args:
        db_url: Database connection URL

    Returns:
        str: Database type ('sqlite', 'postgresql', etc.)
    """
    if db_url.startswith("sqlite"):
        return "sqlite"
    if db_url.startswith("postgresql"):
        return "postgresql"
    # Extract dialect from URL
    return db_url.split(":")[0]


def column_exists(inspector, table_name: str, column_name: str) -> bool:
    """
    Check if a column exists in a table.

    Args:
        inspector: SQLAlchemy inspector object
        table_name: Name of the table to check
        column_name: Name of the column to check

    Returns:
        bool: True if column exists, False otherwise
    """
    try:
        columns = [col["name"] for col in inspector.get_columns(table_name)]
        return column_name in columns
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Error checking if column exists: %s", e, exc_info=True)
        return False


def add_wall_incline_column(
    engine, db_type: str
) -> bool:  # pylint: disable=unused-argument
    """
    Add the wall_incline column to the analyses table.

    Args:
        engine: SQLAlchemy engine
        db_type: Type of database ('sqlite', 'postgresql', etc.)
            Note: Currently unused but kept for API consistency with
            rollback_drop_wall_incline_column() and future database-specific handling.

    Returns:
        bool: True if successful, False otherwise
    """
    inspector = inspect(engine)
    table_name = "analyses"
    column_name = "wall_incline"

    # Check if table exists
    if table_name not in inspector.get_table_names():
        logger.error("Table '%s' does not exist in the database", table_name)
        return False

    # Check if column already exists
    if column_exists(inspector, table_name, column_name):
        logger.info("Column '%s' already exists in '%s' table", column_name, table_name)
        logger.info("Migration already applied")
        return True

    logger.info("Adding '%s' column to '%s' table", column_name, table_name)

    try:
        with engine.begin() as connection:
            # Both SQLite and PostgreSQL support ALTER TABLE ADD COLUMN
            # Note: table_name and column_name are hardcoded constants - do not
            # use user input here as identifiers cannot be parameterized
            sql = text(
                f"ALTER TABLE {table_name} ADD COLUMN {column_name} VARCHAR(20) DEFAULT 'vertical'"
            )
            connection.execute(sql)

        logger.info(
            "Successfully added '%s' column to '%s' table with default 'vertical'",
            column_name,
            table_name,
        )
        return True

    except SQLAlchemyError as e:
        logger.error("Database error while adding column: %s", e, exc_info=True)
        return False
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Unexpected error while adding column: %s", e, exc_info=True)
        return False


def rollback_drop_wall_incline_column(engine, db_type: str) -> bool:
    """
    Rollback: Remove the wall_incline column from the analyses table.

    Args:
        engine: SQLAlchemy engine
        db_type: Type of database ('sqlite', 'postgresql', etc.)

    Returns:
        bool: True if successful, False otherwise
    """
    inspector = inspect(engine)
    table_name = "analyses"
    column_name = "wall_incline"

    # Check if table exists
    if table_name not in inspector.get_table_names():
        logger.error("Table '%s' does not exist in the database", table_name)
        return False

    # Check if column exists
    if not column_exists(inspector, table_name, column_name):
        logger.info("Column '%s' does not exist in '%s' table", column_name, table_name)
        logger.info("Rollback not needed or already applied")
        return True

    logger.info("Removing '%s' column from '%s' table", column_name, table_name)

    try:
        with engine.begin() as connection:
            if db_type == "sqlite":
                # SQLite doesn't support DROP COLUMN in older versions
                # For this rollback, we'll use a simplified approach
                logger.warning(
                    "SQLite detected - dropping column may not be supported in all versions"
                )
                logger.warning(
                    "If this fails, you may need to manually remove the column or restore from backup"
                )
                # Try the modern SQLite syntax (3.35.0+)
                sql = text(f"ALTER TABLE {table_name} DROP COLUMN {column_name}")
            else:
                # PostgreSQL and most other databases support DROP COLUMN
                sql = text(
                    f"ALTER TABLE {table_name} DROP COLUMN IF EXISTS {column_name}"
                )

            connection.execute(sql)

        logger.info(
            "Successfully removed '%s' column from '%s' table", column_name, table_name
        )
        return True

    except SQLAlchemyError as e:
        logger.error("Database error while dropping column: %s", e, exc_info=True)
        logger.error(
            "If using old SQLite version, you may need to manually remove the column"
        )
        return False
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Unexpected error while dropping column: %s", e, exc_info=True)
        return False


def verify_migration(engine) -> bool:
    """
    Verify that the migration was successful.

    Args:
        engine: SQLAlchemy engine

    Returns:
        bool: True if migration is verified successful, False otherwise
    """
    inspector = inspect(engine)
    table_name = "analyses"
    column_name = "wall_incline"

    try:
        # Check that wall_incline column exists
        if not column_exists(inspector, table_name, column_name):
            logger.error("Verification failed: '%s' column does not exist", column_name)
            return False

        # Check that the column has the right type and default
        columns = inspector.get_columns(table_name)
        wall_incline_col = next(
            (col for col in columns if col["name"] == column_name), None
        )

        if wall_incline_col is None:
            logger.error("Could not find column info for '%s'", column_name)
            return False

        logger.info(
            "Column '%s' found with type: %s", column_name, wall_incline_col["type"]
        )
        logger.info("Verification successful")
        return True

    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Verification error: %s", e, exc_info=True)
        return False


def main():  # pylint: disable=too-many-branches,too-many-statements
    """Main migration execution function."""
    parser = argparse.ArgumentParser(
        description="Add wall_incline column to analyses table for Phase 1a grade prediction"
    )
    parser.add_argument(
        "--rollback",
        action="store_true",
        help="Rollback the migration (remove the column)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify if migration is needed or was successful",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check what would be done without making changes",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip interactive confirmation prompts",
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("MIGRATION: Add wall_incline column to analyses table")
    logger.info("=" * 80)

    # Get database URL
    db_url = get_database_url()
    db_type = get_database_type(db_url)

    # Mask password in log output
    safe_db_url = db_url
    try:
        if "@" in db_url:
            # Mask password for PostgreSQL URLs
            parts = db_url.split("@")
            if ":" in parts[0]:
                user_pass = parts[0].split("://")[1]
                if ":" in user_pass:
                    safe_db_url = db_url.replace(user_pass.split(":")[1], "****")
    except (IndexError, ValueError):
        # If URL parsing fails, fall back to showing "[masked]"
        safe_db_url = "[database URL - masked for security]"

    logger.info("Database URL: %s", safe_db_url)
    logger.info("Database Type: %s", db_type)

    # Create engine
    try:
        engine = create_engine(db_url)
        logger.info("Database connection established")
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Failed to connect to database: %s", e, exc_info=True)
        sys.exit(1)

    # Verify-only mode
    if args.verify_only:
        logger.info("Running in verify-only mode")
        inspector = inspect(engine)
        column_exists_now = column_exists(inspector, "analyses", "wall_incline")

        if column_exists_now:
            logger.info(
                "Column 'wall_incline' EXISTS - migration not needed or already applied"
            )
            sys.exit(0)
        else:
            logger.info("Column 'wall_incline' DOES NOT EXIST - migration needed")
            sys.exit(1)

    # Dry-run mode
    if args.dry_run:
        logger.info("Running in dry-run mode - no changes will be made")
        inspector = inspect(engine)
        column_exists_now = column_exists(inspector, "analyses", "wall_incline")

        if args.rollback:
            if column_exists_now:
                logger.info(
                    "DRY RUN: Would drop wall_incline column from analyses table"
                )
            else:
                logger.info("DRY RUN: Column does not exist - rollback not needed")
        else:
            if column_exists_now:
                logger.info("DRY RUN: Column already exists - migration not needed")
            else:
                logger.info(
                    "DRY RUN: Would add wall_incline column to analyses table with default 'vertical'"
                )

        sys.exit(0)

    # Rollback mode
    if args.rollback:
        logger.warning("ROLLBACK MODE: Removing wall_incline column")

        # Prompt for confirmation
        if not args.yes:
            if not sys.stdin.isatty():
                logger.error(
                    "Non-interactive environment detected. Use --yes flag to proceed without confirmation."
                )
                sys.exit(1)
            response = input("Are you sure you want to rollback? (yes/no): ")
            if response.lower() != "yes":
                logger.info("Rollback cancelled by user")
                sys.exit(0)

        success = rollback_drop_wall_incline_column(engine, db_type)

        if success:
            logger.info("Rollback completed successfully")
            sys.exit(0)
        else:
            logger.error("Rollback failed")
            sys.exit(1)

    # Normal migration mode
    logger.info("Starting migration to add wall_incline column")
    logger.info("This migration is safe and uses a default value for existing records")

    # Prompt for confirmation in production
    if os.environ.get("FLASK_ENV") == "production":
        if not args.yes:
            if not sys.stdin.isatty():
                logger.error(
                    "Non-interactive environment detected. Use --yes flag to proceed without confirmation."
                )
                sys.exit(1)
            response = input("Continue with migration? (yes/no): ")
            if response.lower() != "yes":
                logger.info("Migration cancelled by user")
                sys.exit(0)

    # Execute migration
    success = add_wall_incline_column(engine, db_type)

    if not success:
        logger.error("Migration failed")
        sys.exit(1)

    # Verify migration
    logger.info("Verifying migration...")
    if verify_migration(engine):
        logger.info("=" * 80)
        logger.info("MIGRATION COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info("Next steps:")
        logger.info("1. Test your application to ensure it works correctly")
        logger.info("2. Verify that new routes can save wall_incline values")
        logger.info("3. Existing routes will have 'vertical' as the default value")
        sys.exit(0)
    else:
        logger.error("=" * 80)
        logger.error("MIGRATION VERIFICATION FAILED")
        logger.error("=" * 80)
        logger.error("Please check the logs and verify your database state manually")
        sys.exit(1)


if __name__ == "__main__":
    main()
