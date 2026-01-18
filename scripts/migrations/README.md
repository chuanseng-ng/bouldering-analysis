# Database Migrations

This directory contains database migration scripts for the Bouldering Analysis application.

## Overview

Migration scripts in this directory help safely modify the database schema when upgrading the application to new versions. Each migration script is designed to be:

- **Safe**: Includes comprehensive error handling and validation
- **Idempotent**: Can be run multiple times without causing issues
- **Reversible**: Includes rollback functionality when possible
- **Well-documented**: Contains clear instructions and logging

## Available Migrations

### drop_holds_detected_column.py

Removes the deprecated `holds_detected` JSON column from the `analyses` table.

**When to run:** After upgrading from a version that stored hold detections in a JSON column to the new version that uses the `DetectedHold` relationship table.

**Documentation:** See [`docs/migrations.md`](../../docs/migrations.md) for detailed instructions.

**Usage:**

```bash
# Dry run (check what would happen)
python scripts/migrations/drop_holds_detected_column.py --dry-run

# Run the migration
python scripts/migrations/drop_holds_detected_column.py

# Verify migration status
python scripts/migrations/drop_holds_detected_column.py --verify-only

# Rollback (re-add empty column)
python scripts/migrations/drop_holds_detected_column.py --rollback
```

## Running Migrations

### Prerequisites

1. **Always backup your database first!**

   ```bash
   # SQLite
   cp bouldering_analysis.db bouldering_analysis.db.backup

   # PostgreSQL
   pg_dump -U username -d database_name > backup.sql
   ```

2. Stop the application before running migrations

3. Ensure you have the necessary database permissions

### General Process

1. **Review the migration documentation** in [`docs/migrations.md`](../../docs/migrations.md)
2. **Run in dry-run mode** to see what will change
3. **Backup your database**
4. **Run the migration**
5. **Verify the results**
6. **Test your application**

### Database Configuration

Migration scripts use the same database configuration as the main application:

- `DATABASE_URL` environment variable (or defaults to `sqlite:///bouldering_analysis.db`)
- Supports both SQLite and PostgreSQL

## Writing New Migrations

When creating new migration scripts, follow these guidelines:

### Structure

```python
#!/usr/bin/env python3
"""
Brief description of what this migration does.

WHEN TO RUN:
- Explain when this migration should be run

PREREQUISITES:
- List what needs to be in place before running

HOW TO VERIFY SUCCESS:
- Explain how to verify the migration worked

HOW TO ROLLBACK:
- Explain how to undo the migration if needed
"""

# Standard imports
import os
import sys
import logging
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# SQLAlchemy imports
from sqlalchemy import create_engine, inspect, text  # noqa: E402
from sqlalchemy.exc import SQLAlchemyError  # noqa: E402

# Your migration code here
```

### Required Features

Every migration script should include:

1. **Comprehensive documentation** in docstring and code comments
2. **Database connection** using the application's configuration pattern
3. **Idempotency checking** - detect if migration is already applied
4. **Both SQLite and PostgreSQL support** (or clearly document limitations)
5. **Error handling** with proper logging
6. **Verification** to confirm the migration succeeded
7. **Rollback capability** when feasible
8. **Command-line options**:
   - `--dry-run`: Show what would happen without making changes
   - `--verify-only`: Check if migration is needed
   - `--rollback`: Undo the migration
9. **Logging** to both console and log file
10. **User confirmation** for production environments

### Testing

Before committing a migration script:

1. Test with a copy of a production database
2. Verify it works with both SQLite and PostgreSQL
3. Test all command-line options (dry-run, verify-only, rollback)
4. Verify the rollback functionality
5. Ensure logging is comprehensive and clear
6. Test idempotency (running twice should be safe)

## Troubleshooting

If a migration fails:

1. **Check the log file** in `logs/migration_*.log`
2. **Review the error message** for specific issues
3. **Restore from backup** if necessary
4. **Consult the documentation** in [`docs/migrations.md`](../../docs/migrations.md)

## Best Practices

- **Always backup before migrating**
- **Test migrations on a copy of production data first**
- **Run migrations during low-traffic periods**
- **Keep backups for at least a week after migration**
- **Verify application functionality after migration**
- **Document any issues or unusual circumstances**

## Need Help?

For detailed migration instructions and troubleshooting, see [`docs/migrations.md`](../../docs/migrations.md).
