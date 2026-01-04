#!/usr/bin/env python3
"""
Migration script to drop the deprecated holds_detected column from the analyses table.

WHEN TO RUN:
- Run this migration after upgrading to a version that uses the DetectedHold relationship table
- Only affects existing databases that have the old holds_detected JSON column
- New installations won't have this column and can skip this migration

PREREQUISITES:
- All hold detection data should already be migrated to the DetectedHold table
- Ensure you have a database backup before running this migration

HOW TO VERIFY SUCCESS:
- Query the analyses table schema to confirm holds_detected column is removed
- Verify DetectedHold table contains all hold detection data
- Check that the application functions normally after migration

HOW TO ROLLBACK:
- Run this script with the --rollback flag to re-add the column (will be empty)
- Restore from backup if you need to recover the original data
"""

from __future__ import annotations

import os
import sys
import logging
import argparse
from pathlib import Path

# Add the project root to the Python path to allow imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import create_engine, inspect, text  # noqa: E402
from sqlalchemy.exc import SQLAlchemyError  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

# Ensure logs directory exists before configuring FileHandler
project_root.joinpath("logs").mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            project_root / "logs" / "migration_drop_holds_detected.log"
        ),
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
    else:
        if db_url.startswith("postgresql"):
            return "postgresql"
        else:
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
    except Exception as e:
        logger.error("Error checking if column exists: %s", e, exc_info=True)
        return False


def drop_holds_detected_column(engine, db_type: str) -> bool:
    """
    Drop the holds_detected column from the analyses table.

    Args:
        engine: SQLAlchemy engine
        db_type: Type of database ('sqlite', 'postgresql', etc.)

    Returns:
        bool: True if successful, False otherwise
    """
    inspector = inspect(engine)
    table_name = "analyses"
    column_name = "holds_detected"

    # Check if table exists
    if table_name not in inspector.get_table_names():
        logger.error("Table '%s' does not exist in the database", table_name)
        return False

    # Check if column exists
    if not column_exists(inspector, table_name, column_name):
        logger.info("Column '%s' does not exist in '%s' table", column_name, table_name)
        logger.info("Migration already applied or column never existed")
        return True

    logger.info("Found '%s' column in '%s' table", column_name, table_name)

    try:
        with engine.begin() as connection:
            if db_type == "sqlite":
                # SQLite doesn't support DROP COLUMN directly
                # We need to create a new table without the column and copy data
                logger.info("SQLite detected - using table recreation method")

                # Get current table columns
                columns = inspector.get_columns(table_name)
                new_columns = [col for col in columns if col["name"] != column_name]
                column_names = [col["name"] for col in new_columns]
                column_names_str = ", ".join(column_names)

                # Get indexes
                indexes = inspector.get_indexes(table_name)

                # Start transaction
                logger.info("Creating temporary table without holds_detected column")

                # Capture current foreign key state
                fk_result = connection.execute(text("PRAGMA foreign_keys"))
                fk_state = fk_result.scalar()
                logger.info("Current foreign key state: %s", fk_state)

                # Disable foreign keys for table recreation
                connection.execute(text("PRAGMA foreign_keys=OFF"))
                logger.info("Foreign keys disabled for table recreation")

                try:
                    # Create temporary table with all columns except holds_detected
                    # Note: We're doing this the simple way by copying data
                    connection.execute(
                        text(f"ALTER TABLE {table_name} RENAME TO {table_name}_old")
                    )

                    # Create new table (let SQLAlchemy handle schema creation is complex)
                    # Instead, we'll use raw SQL based on what we know about the table
                    create_table_sql = f"""
                    CREATE TABLE {table_name} (
                        id VARCHAR(36) PRIMARY KEY,
                        image_filename VARCHAR(255) NOT NULL,
                        image_path VARCHAR(500) NOT NULL,
                        predicted_grade VARCHAR(10) NOT NULL,
                        confidence_score FLOAT,
                        features_extracted JSON,
                        created_at TIMESTAMP,
                        updated_at TIMESTAMP
                    )
                    """
                    connection.execute(text(create_table_sql))

                    # Copy data from old table
                    logger.info("Copying data from old table to new table")
                    copy_sql = f"""
                    INSERT INTO {table_name} ({column_names_str})
                    SELECT {column_names_str} FROM {table_name}_old
                    """
                    connection.execute(text(copy_sql))

                    # Recreate indexes
                    for index in indexes:
                        if index["name"] and index["column_names"]:
                            cols = ", ".join(index["column_names"])
                            unique = "UNIQUE" if index["unique"] else ""
                            index_sql = f"CREATE {unique} INDEX {index['name']} ON {table_name} ({cols})"
                            try:
                                connection.execute(text(index_sql))
                            except Exception as e:
                                logger.warning(
                                    "Could not recreate index %s: %s",
                                    index["name"],
                                    e,
                                    exc_info=True,
                                )

                    # Drop old table
                    logger.info("Dropping old table")
                    connection.execute(text(f"DROP TABLE {table_name}_old"))

                finally:
                    # Always restore foreign key state
                    if fk_state:
                        connection.execute(text("PRAGMA foreign_keys=ON"))
                        logger.info("Foreign keys restored to ON")
                    else:
                        connection.execute(text("PRAGMA foreign_keys=OFF"))
                        logger.info("Foreign keys restored to OFF")

            else:
                # PostgreSQL and most other databases support ALTER TABLE DROP COLUMN
                logger.info(
                    f"{db_type.upper()} detected - using ALTER TABLE DROP COLUMN"
                )
                sql = text(
                    f"ALTER TABLE {table_name} DROP COLUMN IF EXISTS {column_name}"
                )
                connection.execute(sql)

        logger.info(
            "Successfully dropped '%s' column from '%s' table", column_name, table_name
        )
        return True

    except SQLAlchemyError as e:
        logger.error("Database error while dropping column: %s", e, exc_info=True)
        return False
    except Exception as e:
        logger.error("Unexpected error while dropping column: %s", e, exc_info=True)
        return False


def rollback_add_holds_detected_column(engine, db_type: str) -> bool:
    """
    Rollback: Add the holds_detected column back to the analyses table.

    Note: This will add an empty column. The original data cannot be recovered
    unless you restore from a backup.

    Args:
        engine: SQLAlchemy engine
        db_type: Type of database ('sqlite', 'postgresql', etc.)

    Returns:
        bool: True if successful, False otherwise
    """
    inspector = inspect(engine)
    table_name = "analyses"
    column_name = "holds_detected"

    # Check if table exists
    if table_name not in inspector.get_table_names():
        logger.error("Table '%s' does not exist in the database", table_name)
        return False

    # Check if column already exists
    if column_exists(inspector, table_name, column_name):
        logger.info("Column '%s' already exists in '%s' table", column_name, table_name)
        logger.info("Rollback already applied or no migration to rollback")
        return True

    logger.info("Adding '%s' column to '%s' table", column_name, table_name)

    try:
        with engine.begin() as connection:
            if db_type == "sqlite":
                # SQLite supports ALTER TABLE ADD COLUMN
                sql = text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} JSON")
            elif db_type == "postgresql":
                sql = text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} JSONB")
            else:
                # Generic JSON type for other databases
                sql = text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} JSON")

            connection.execute(sql)

        logger.info(
            "Successfully added '%s' column to '%s' table", column_name, table_name
        )
        logger.warning(
            "Column added but is EMPTY - restore from backup to recover data"
        )
        return True

    except SQLAlchemyError as e:
        logger.error("Database error while adding column: %s", e, exc_info=True)
        return False
    except Exception as e:
        logger.error("Unexpected error while adding column: %s", e, exc_info=True)
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
    column_name = "holds_detected"

    try:
        # Check that holds_detected column is gone
        column_exists_now = column_exists(inspector, table_name, column_name)

        if column_exists_now:
            logger.error("Verification failed: '%s' column still exists", column_name)
            return False

        # Check that DetectedHold table exists
        if "detected_holds" not in inspector.get_table_names():
            logger.warning(
                "DetectedHold table does not exist - may need to run db.create_all()"
            )
            return False

        # Check that the analyses table is still accessible
        Session = sessionmaker(bind=engine)
        session = Session()
        try:
            result = session.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
            count = result.scalar()
            logger.info(
                "Verification successful: %s records in %s table", count, table_name
            )
            return True
        finally:
            session.close()

    except Exception as e:
        logger.error("Verification error: %s", e, exc_info=True)
        return False


def main():
    """Main migration execution function."""
    parser = argparse.ArgumentParser(
        description="Drop the deprecated holds_detected column from analyses table"
    )
    parser.add_argument(
        "--rollback",
        action="store_true",
        help="Rollback the migration (re-add the column as empty)",
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
    logger.info("MIGRATION: Drop holds_detected column from analyses table")
    logger.info("=" * 80)

    # Get database URL
    db_url = get_database_url()
    db_type = get_database_type(db_url)

    # Mask password in log output
    safe_db_url = db_url
    if "@" in db_url:
        # Mask password for PostgreSQL URLs
        parts = db_url.split("@")
        if ":" in parts[0]:
            user_pass = parts[0].split("://")[1]
            if ":" in user_pass:
                safe_db_url = db_url.replace(user_pass.split(":")[1], "****")

    logger.info("Database URL: %s", safe_db_url)
    logger.info("Database Type: %s", db_type)

    # Create engine
    try:
        engine = create_engine(db_url)
        logger.info("Database connection established")
    except Exception as e:
        logger.error("Failed to connect to database: %s", e, exc_info=True)
        sys.exit(1)

    # Verify-only mode
    if args.verify_only:
        logger.info("Running in verify-only mode")
        inspector = inspect(engine)
        column_exists_now = column_exists(inspector, "analyses", "holds_detected")

        if column_exists_now:
            logger.info("Column 'holds_detected' EXISTS - migration needed")
            sys.exit(1)
        else:
            logger.info(
                "Column 'holds_detected' DOES NOT EXIST - migration not needed or already applied"
            )
            sys.exit(0)

    # Dry-run mode
    if args.dry_run:
        logger.info("Running in dry-run mode - no changes will be made")
        inspector = inspect(engine)
        column_exists_now = column_exists(inspector, "analyses", "holds_detected")

        if args.rollback:
            if column_exists_now:
                logger.info("DRY RUN: Column already exists - rollback not needed")
            else:
                logger.info(
                    "DRY RUN: Would add holds_detected column to analyses table"
                )
        else:
            if column_exists_now:
                logger.info(
                    "DRY RUN: Would drop holds_detected column from analyses table"
                )
            else:
                logger.info("DRY RUN: Column does not exist - migration not needed")

        sys.exit(0)

    # Rollback mode
    if args.rollback:
        logger.warning("ROLLBACK MODE: Re-adding holds_detected column")
        logger.warning(
            "This will create an EMPTY column - restore from backup to recover data"
        )

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

        success = rollback_add_holds_detected_column(engine, db_type)

        if success:
            logger.info("Rollback completed successfully")
            sys.exit(0)
        else:
            logger.error("Rollback failed")
            sys.exit(1)

    # Normal migration mode
    logger.info("Starting migration to drop holds_detected column")
    logger.warning("Make sure you have backed up your database!")

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
    success = drop_holds_detected_column(engine, db_type)

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
        logger.info("2. Verify that DetectedHold table contains all hold data")
        logger.info("3. Keep your database backup for at least a few days")
        sys.exit(0)
    else:
        logger.error("=" * 80)
        logger.error("MIGRATION VERIFICATION FAILED")
        logger.error("=" * 80)
        logger.error("Please check the logs and verify your database state manually")
        sys.exit(1)


if __name__ == "__main__":
    main()
