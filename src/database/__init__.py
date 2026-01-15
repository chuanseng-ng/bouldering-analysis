"""Database layer for Supabase integration.

This package provides Supabase client management and database operations.
"""

from src.database.supabase_client import get_supabase_client

__all__ = ["get_supabase_client"]
