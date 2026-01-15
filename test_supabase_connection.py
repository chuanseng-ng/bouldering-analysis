"""Test Supabase connection and configuration.

This script verifies that your Supabase credentials are configured correctly
and that you can connect to your Supabase project.

Usage:
    python test_supabase_connection.py
"""

import sys
from typing import Any

from src.database import get_supabase_client
from src.database.supabase_client import SupabaseClientError


def test_connection() -> bool:
    """Test basic Supabase connection.

    Returns:
        True if connection succeeds, False otherwise.
    """
    try:
        client = get_supabase_client()
        print("[OK] Successfully connected to Supabase!")
        print(f"     Project URL: {client.supabase_url}")
        return True
    except SupabaseClientError as e:
        print(f"[FAIL] Connection failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check that BA_SUPABASE_URL is set in your .env file")
        print("2. Check that BA_SUPABASE_KEY is set in your .env file")
        print("3. Verify your Supabase project is active")
        return False
    except Exception as e:
        print(f"[FAIL] Unexpected error: {e}")
        return False


def test_storage() -> bool:
    """Test Supabase Storage access.

    Returns:
        True if storage is accessible, False otherwise.
    """
    try:
        client = get_supabase_client()

        # Try to list all buckets (requires admin permissions)
        try:
            buckets = client.storage.list_buckets()
            if buckets:
                print("\n[OK] Storage access successful!")
                print(f"     Found {len(buckets)} storage bucket(s):")
                for bucket in buckets:
                    # Extract bucket details
                    bucket_dict = bucket.__dict__ if hasattr(bucket, '__dict__') else bucket
                    if isinstance(bucket_dict, dict):
                        bucket_name = bucket_dict.get("name", "unknown")
                        is_public = bucket_dict.get("public", False)
                    else:
                        bucket_name = getattr(bucket, 'name', 'unknown')
                        is_public = getattr(bucket, 'public', False)
                    visibility = "public" if is_public else "private"
                    print(f"       - {bucket_name} ({visibility})")
                return True
        except Exception:
            pass  # Fall through to direct bucket check

        # If list_buckets returns empty or fails, try direct bucket access
        # This is normal for anon keys (they can't list all buckets)
        print("\n[INFO] Checking storage buckets directly...")
        print("       (anon key can't list all buckets - this is normal)")

        required_buckets = ["route-images", "model-outputs"]
        found_buckets = []
        missing_buckets = []

        for bucket_name in required_buckets:
            try:
                # Try to access the bucket
                bucket_proxy = client.storage.from_(bucket_name)
                bucket_proxy.list()  # This will raise if bucket doesn't exist
                found_buckets.append(bucket_name)
                print(f"       [OK] {bucket_name} - accessible")
            except Exception as e:
                error_msg = str(e).lower()
                if "not found" in error_msg or "does not exist" in error_msg:
                    missing_buckets.append(bucket_name)
                    print(f"       [MISSING] {bucket_name} - not found")
                else:
                    print(f"       [ERROR] {bucket_name} - {e}")

        if found_buckets:
            print(
                f"\n[OK] Found {len(found_buckets)} bucket(s): {', '.join(found_buckets)}"
            )

        if missing_buckets:
            print(f"\n[WARN] Missing buckets: {', '.join(missing_buckets)}")
            print("       Create them in: Settings -> Storage -> New bucket")
            print("       See docs/SUPABASE_SETUP.md Step 4 for instructions")

        return len(found_buckets) > 0

    except Exception as e:
        print(f"\n[FAIL] Storage access failed: {e}")
        print("\nThis may be due to:")
        print("1. API key permissions (ensure you're using anon or service_role key)")
        print("2. Network connectivity issues")
        print("3. Supabase project configuration")
        return False


def main() -> int:
    """Run all Supabase connection tests.

    Returns:
        0 if all tests pass, 1 otherwise.
    """
    print("=" * 60)
    print("Testing Supabase Connection")
    print("=" * 60)

    # Test connection
    print("\n[1/2] Testing Supabase connection...")
    connection_ok = test_connection()

    if not connection_ok:
        print("\n" + "=" * 60)
        print("[FAIL] Connection test failed - skipping remaining tests")
        print("=" * 60)
        return 1

    # Test storage
    print("\n[2/2] Testing Supabase Storage...")
    storage_ok = test_storage()

    # Summary
    print("\n" + "=" * 60)
    if connection_ok and storage_ok:
        print("[SUCCESS] All tests passed! Your Supabase setup is working.")
    else:
        print("[WARNING] Some tests failed. See error messages above.")
    print("=" * 60)

    return 0 if (connection_ok and storage_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
