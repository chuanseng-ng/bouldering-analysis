"""Shared utilities for bouldering-analysis database migration scripts.

Provides the project-root path, logging setup, and common database helpers
used by every migration script to avoid code duplication.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from urllib.parse import urlparse, urlunparse

from sqlalchemy.engine import Inspector

# ── project root (scripts/migrations/ → 3 levels up) ────────────────────────
project_root: Path = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

_logger = logging.getLogger(__name__)


def setup_migration_logging(log_filename: str) -> None:
    """Configure root logging for a migration script.

    Creates the logs directory if it does not exist and calls
    ``logging.basicConfig`` with a StreamHandler (stdout) and a FileHandler.
    Callers should obtain their own logger via ``logging.getLogger(__name__)``
    after this call.

    Args:
        log_filename: Name of the log file (e.g. ``"migration_foo.log"``).
    """
    project_root.joinpath("logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(project_root / "logs" / log_filename),
        ],
    )


def get_database_url() -> str:
    """Get the database URL from environment or use default.

    Uses the same configuration pattern as src/main.py.

    Returns:
        Database connection URL.
    """
    db_url = os.environ.get("DATABASE_URL") or "sqlite:///bouldering_analysis.db"

    # Convert relative SQLite paths to absolute paths from project root.
    if db_url.startswith("sqlite:///") and not db_url.startswith("sqlite:////"):
        db_path = db_url.replace("sqlite:///", "")
        if not os.path.isabs(db_path):
            db_path = str(project_root / db_path)
            db_url = f"sqlite:///{db_path}"

    return db_url


def get_database_type(db_url: str) -> str:
    """Determine the database type from the connection URL.

    Args:
        db_url: Database connection URL.

    Returns:
        Database type (``'sqlite'``, ``'postgresql'``, etc.).
    """
    if db_url.startswith("sqlite"):
        return "sqlite"
    if db_url.startswith("postgresql"):
        return "postgresql"
    # Extract dialect from URL.
    return db_url.split(":")[0]


def mask_db_url(db_url: str) -> str:
    """Mask the password in a database URL for safe logging.

    Uses ``urllib.parse.urlparse`` for robust URL handling rather than
    fragile string splitting.

    Args:
        db_url: Database connection URL.

    Returns:
        URL with password replaced by ``"****"``, or a fallback string if
        parsing fails.
    """
    try:
        parsed = urlparse(db_url)
        if not parsed.password:
            return db_url
        userinfo = f"{parsed.username}:****"
        host = parsed.hostname or ""
        netloc = (
            f"{userinfo}@{host}:{parsed.port}" if parsed.port else f"{userinfo}@{host}"
        )
        return urlunparse(parsed._replace(netloc=netloc))
    except (ValueError, AttributeError):
        return "[database URL - masked for security]"


def column_exists(inspector: Inspector, table_name: str, column_name: str) -> bool:
    """Check if a column exists in a table.

    Args:
        inspector: SQLAlchemy inspector object.
        table_name: Name of the table to check.
        column_name: Name of the column to check.

    Returns:
        True if column exists, False otherwise.
    """
    try:
        columns = [col["name"] for col in inspector.get_columns(table_name)]
        return column_name in columns
    except Exception as e:  # pylint: disable=broad-exception-caught
        _logger.error("Error checking if column exists: %s", e, exc_info=True)
        return False
