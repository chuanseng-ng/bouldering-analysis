#!/usr/bin/env python3
"""Keep-alive ping script for the Supabase-backed bouldering-analysis backend.

Supabase pauses free-tier projects after approximately 1 week of inactivity.
Run this script periodically (e.g., via GitHub Actions or a cron job) to keep
the database active by hitting the health endpoint.

Usage:
    python scripts/ping_supabase.py --url https://your-backend.com
    python scripts/ping_supabase.py          # reads BA_APP_BASE_URL env var

Exit codes:
    0  — HTTP 200 OK received (database is alive)
    1  — Any error (connection refused, non-200 status, timeout, etc.)

GitHub Actions example (every Monday at noon UTC):
    on:
      schedule:
        - cron: '0 12 * * 1'
    steps:
      - run: python scripts/ping_supabase.py --url ${{ secrets.BACKEND_URL }}
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

_HEALTH_PATH = "/api/v1/health/db"
_TIMEOUT_SECONDS = 15


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Returns:
        Configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        description="Ping the Supabase-backed backend to prevent free-tier pauses.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/ping_supabase.py --url https://your-backend.com
  BA_APP_BASE_URL=https://your-backend.com python scripts/ping_supabase.py
""",
    )
    parser.add_argument(
        "--url",
        metavar="BASE_URL",
        default=os.environ.get("BA_APP_BASE_URL", ""),
        help=(
            "Base URL of the deployed backend "
            "(e.g. https://your-backend.com). "
            "Falls back to BA_APP_BASE_URL environment variable."
        ),
    )
    return parser


def ping(base_url: str) -> int:
    """Send a GET request to the health endpoint and return the HTTP status code.

    Args:
        base_url: Base URL of the backend (e.g. ``"https://your-backend.com"``).

    Returns:
        HTTP status code received from the server.

    Raises:
        urllib.error.URLError: If the connection could not be established.
        ValueError: If ``base_url`` is empty.
    """
    if not base_url:
        raise ValueError("Base URL is required. Pass --url or set BA_APP_BASE_URL.")

    # Validate URL scheme to prevent non-HTTP requests (SSRF / local file reads)
    parsed = urllib.parse.urlparse(base_url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"URL scheme must be 'http' or 'https', got: {parsed.scheme!r}"
        )

    # Strip trailing slash for a clean URL
    url = base_url.rstrip("/") + _HEALTH_PATH

    request = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:  # noqa: S310
        status: int = response.status
    return status


def main() -> None:
    """Entry point for the keep-alive ping script.

    Exits with code 0 on HTTP 200, 1 on any failure.
    """
    parser = _build_parser()
    args = parser.parse_args()

    try:
        status_code = ping(args.url)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    except urllib.error.HTTPError as exc:
        print(
            f"PING FAILED — HTTP {exc.code} {exc.reason} — {args.url}{_HEALTH_PATH}",
            file=sys.stderr,
        )
        sys.exit(1)
    except urllib.error.URLError as exc:
        print(
            f"PING FAILED — connection error: {exc.reason} — {args.url}{_HEALTH_PATH}",
            file=sys.stderr,
        )
        sys.exit(1)
    except TimeoutError:
        print(
            f"PING FAILED — timed out after {_TIMEOUT_SECONDS}s — {args.url}{_HEALTH_PATH}",
            file=sys.stderr,
        )
        sys.exit(1)

    if status_code == 200:
        print(f"PING OK — HTTP {status_code} — {args.url}{_HEALTH_PATH}")
        sys.exit(0)
    else:
        print(
            f"PING FAILED — HTTP {status_code} — {args.url}{_HEALTH_PATH}",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
