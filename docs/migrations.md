# Database Migrations

This document describes database migrations for the Bouldering Analysis application and provides instructions for running them safely.

## Table of Contents

- [Overview](#overview)
- [Migration: Drop holds_detected Column](#migration-drop-holds_detected-column)
  - [Background](#background)
  - [Prerequisites](#prerequisites)
  - [Running the Migration](#running-the-migration)
  - [Verification](#verification)
  - [Rollback](#rollback)
  - [Troubleshooting](#troubleshooting)

## Overview

Database migrations are necessary when the application's data model changes. This document covers migrations that may need to be applied to existing databases when upgrading the application.

**Important Notes:**

- New installations do NOT need to run these migrations - the database schema will be created correctly from the start
- Always backup your database before running any migration
- Migrations are designed to be idempotent (safe to run multiple times)

## Migration: Drop holds_detected Column

### Background

**What changed:**

The `Analysis` model previously stored hold detection results in a `holds_detected` JSON column. This approach had several limitations:
- Difficult to query and filter individual holds
- No foreign key relationships for data integrity
- Limited ability to aggregate statistics about holds
- Challenging to perform analytics on hold types and distributions

**The new approach:**
Hold detection results are now stored in a dedicated `DetectedHold` relationship table with the following benefits:

- Individual holds are stored as separate records with foreign key relationships
- Easy to query holds by type, confidence, or analysis
- Better data integrity through foreign key constraints
- Improved performance for hold-related queries
- Enables advanced analytics and statistics

**Who needs this migration:**

- Only users upgrading from a version that used the `holds_detected` JSON column
- New installations will not have this column and do not need this migration

### Prerequisites

Before running this migration, ensure:

1. **You have a database backup**
2. 
   ```bash
   # For SQLite
   cp bouldering_analysis.db bouldering_analysis.db.backup
   
   # For PostgreSQL
   pg_dump -U username -d database_name > backup.sql
   ```

3. **The DetectedHold table exists and contains data**

   The migration assumes that hold data has already been migrated from the JSON column to the `DetectedHold` table. Verify this by checking:

   ```python
   from src.models import db, DetectedHold, Analysis
   from src.main import app
   
   with app.app_context():
       # Check that DetectedHold table has data
       hold_count = db.session.query(DetectedHold).count()
       analysis_count = db.session.query(Analysis).count()
       print(f"Analyses: {analysis_count}, DetectedHolds: {hold_count}")
       
       # Ideally, you should have multiple holds per analysis
       if analysis_count > 0 and hold_count == 0:
           print("WARNING: No holds found in DetectedHold table!")
   ```

4. **The application is not running**

   Stop your Flask application before running the migration to avoid database lock issues.

### Running the Migration

The migration script is located at [`scripts/migrations/drop_holds_detected_column.py`](../scripts/migrations/drop_holds_detected_column.py).

#### Step 1: Review what will be changed (Dry Run)

First, run the migration in dry-run mode to see what would happen without making any changes:

```bash
python scripts/migrations/drop_holds_detected_column.py --dry-run
```

This will show you:

- Whether the `holds_detected` column exists
- What action would be taken
- No actual changes will be made

#### Step 2: Run the migration

Execute the migration:

```bash
python scripts/migrations/drop_holds_detected_column.py
```

**For production environments**, the script will prompt for confirmation before proceeding.

The script will:

1. Connect to your database using the same configuration as the main application (reads `DATABASE_URL` environment variable or defaults to SQLite)
2. Check if the `holds_detected` column exists in the `analyses` table
3. Drop the column using the appropriate method for your database type:
   - **SQLite**: Creates a new table without the column, copies data, and replaces the old table
   - **PostgreSQL**: Uses `ALTER TABLE DROP COLUMN`
4. Verify the migration was successful
5. Log all actions to `logs/migration_drop_holds_detected.log`

**Example output:**

```text
2026-01-04 09:30:00 - INFO - ================================================================================
2026-01-04 09:30:00 - INFO - MIGRATION: Drop holds_detected column from analyses table
2026-01-04 09:30:00 - INFO - ================================================================================
2026-01-04 09:30:00 - INFO - Database URL: sqlite:///c:/path/to/bouldering_analysis.db
2026-01-04 09:30:00 - INFO - Database Type: sqlite
2026-01-04 09:30:00 - INFO - Database connection established
2026-01-04 09:30:00 - INFO - Starting migration to drop holds_detected column
2026-01-04 09:30:00 - INFO - Found 'holds_detected' column in 'analyses' table
2026-01-04 09:30:00 - INFO - SQLite detected - using table recreation method
2026-01-04 09:30:01 - INFO - Creating temporary table without holds_detected column
2026-01-04 09:30:01 - INFO - Copying data from old table to new table
2026-01-04 09:30:01 - INFO - Dropping old table
2026-01-04 09:30:01 - INFO - Successfully dropped 'holds_detected' column from 'analyses' table
2026-01-04 09:30:01 - INFO - Verifying migration...
2026-01-04 09:30:01 - INFO - Verification successful: 42 records in analyses table
2026-01-04 09:30:01 - INFO - ================================================================================
2026-01-04 09:30:01 - INFO - MIGRATION COMPLETED SUCCESSFULLY
2026-01-04 09:30:01 - INFO - ================================================================================
```

#### Environment Variables

The script uses the same database configuration as the main application:

- `DATABASE_URL`: Database connection string (default: `sqlite:///bouldering_analysis.db`)
- `FLASK_ENV`: If set to `production`, will prompt for confirmation before making changes

**Examples:**

```bash
# Using default SQLite database
python scripts/migrations/drop_holds_detected_column.py

# Using a specific SQLite database
DATABASE_URL="sqlite:///path/to/my_database.db" python scripts/migrations/drop_holds_detected_column.py

# Using PostgreSQL
DATABASE_URL="postgresql://user:password@localhost/bouldering_db" python scripts/migrations/drop_holds_detected_column.py
```

### Verification

After running the migration, verify it was successful:

#### Option 1: Use the verify-only flag

```bash
python scripts/migrations/drop_holds_detected_column.py --verify-only
```

This will check if the column exists and exit with status code:

- `0` if column does not exist (migration successful or not needed)
- `1` if column still exists (migration needed or failed)

#### Option 2: Manual verification

**Check the database schema:**

For SQLite:

```bash
sqlite3 bouldering_analysis.db ".schema analyses"
```

For PostgreSQL:

```sql
\d analyses
```

The `holds_detected` column should NOT appear in the schema.

**Verify DetectedHold table has data:**

```python
from src.models import db, DetectedHold
from src.main import app

with app.app_context():
    hold_count = db.session.query(DetectedHold).count()
    print(f"Total detected holds: {hold_count}")
    
    # Check a sample
    sample_holds = db.session.query(DetectedHold).limit(5).all()
    for hold in sample_holds:
        print(f"Hold {hold.id}: Type {hold.hold_type_id}, Confidence {hold.confidence}")
```

**Test the application:**

1. Start your Flask application
2. Upload a test image
3. Verify that hold detection works correctly
4. Check that the analysis results are displayed properly
5. Confirm no errors appear in the logs

### Rollback

If you need to rollback the migration (re-add the column), use the `--rollback` flag:

```bash
python scripts/migrations/drop_holds_detected_column.py --rollback
```

**Important warnings:**

- This will re-add the `holds_detected` column, but it will be **EMPTY**
- The original JSON data **cannot be recovered** unless you restore from a backup
- You will be prompted for confirmation before proceeding

**When to rollback:**

- If you need to revert to an older version of the application that requires the `holds_detected` column
- If you discover issues after migration and need to restore from backup

**Rollback process:**

1. The script adds the `holds_detected` column back to the `analyses` table
2. The column will be `NULL` for all existing records
3. You must restore from backup to recover the original data

**To fully restore from backup:**

For SQLite:

```bash
# Stop the application first
cp bouldering_analysis.db.backup bouldering_analysis.db
```

For PostgreSQL:

```bash
# Stop the application first
psql -U username -d database_name < backup.sql
```

### Troubleshooting

#### Issue: "Column does not exist"

If you see this message, it means:

- The migration has already been run, or
- The column never existed in your database (new installation)

**Action:** No action needed. The migration is not necessary.

#### Issue: "Table 'analyses' does not exist"

This means your database hasn't been initialized yet.

**Action:** Run the application once to create the database tables:

```python
from src.main import app, create_tables

with app.app_context():
    create_tables()
```

#### Issue: "Database connection failed"

Check that:

1. Your `DATABASE_URL` environment variable is set correctly
2. For PostgreSQL, the database server is running
3. You have the necessary permissions to access the database
4. For SQLite, the database file path is correct and writable

#### Issue: Migration appears successful but application crashes

**Possible causes:**

1. Application code still references the old `holds_detected` column
2. DetectedHold table is missing data

**Actions:**

1. Check application logs for specific error messages
2. Verify DetectedHold table has the expected data (see Verification section)
3. If necessary, rollback and restore from backup
4. Report the issue with full error logs

#### Issue: "Permission denied" when running migration

**For SQLite:**

- Ensure you have write permissions for the database file and directory
- Check that no other process has locked the database file

**For PostgreSQL:**

- Ensure your database user has ALTER TABLE permissions
- You may need to run as a superuser for schema modifications

#### Issue: Performance problems during migration

**For large databases:**

- The SQLite migration recreates the entire table, which can be slow
- Consider scheduling the migration during off-peak hours
- The migration time is roughly proportional to the number of Analysis records

**Progress monitoring:**

- Watch the log file in real-time: `tail -f logs/migration_drop_holds_detected.log`
- The script logs each major step

## Need Help?

If you encounter issues not covered in this document:

1. Check the migration log file at `logs/migration_drop_holds_detected.log`
2. Review the script source code at [`scripts/migrations/drop_holds_detected_column.py`](../scripts/migrations/drop_holds_detected_column.py)
3. Ensure you have a recent database backup before attempting fixes
4. If in doubt, restore from backup and seek assistance

## Future Migrations

As the application evolves, additional migrations may be added to this directory. Each migration will be documented in this file with:

- Background on why the migration is needed
- Step-by-step instructions
- Verification procedures
- Rollback instructions

Always check this documentation before upgrading to a new version of the application.
