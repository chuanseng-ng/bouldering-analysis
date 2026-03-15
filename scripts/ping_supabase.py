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
import json
import logging
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

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

    Parses the JSON response body and raises ``RuntimeError`` if the service
    reports a non-healthy status (e.g. ``"degraded"`` or ``"unhealthy"``), so
    that degraded database states are treated as failures even when the HTTP
    response is 200.

    Args:
        base_url: Base URL of the backend (e.g. ``"https://your-backend.com"``).

    Returns:
        HTTP status code received from the server.

    Raises:
        urllib.error.URLError: If the connection could not be established.
        ValueError: If ``base_url`` is empty or has an invalid scheme.
        RuntimeError: If the response JSON indicates a non-healthy service status.
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
        http_status: int = response.status
        body: dict = json.loads(response.read().decode())

    service_status = body.get("status")
    if service_status != "healthy":
        raise RuntimeError(f"service status is '{service_status}'")

    return http_status


def main() -> None:
    """Entry point for the keep-alive ping script.

    Exits with code 0 on HTTP 200, 1 on any failure.
    """
    parser = _build_parser()
    args = parser.parse_args()

    try:
        status_code = ping(args.url)
    except ValueError as exc:
        logger.error("%s", exc)
        sys.exit(1)
    except RuntimeError as exc:
        logger.error("PING FAILED — %s — %s%s", exc, args.url, _HEALTH_PATH)
        sys.exit(1)
    except urllib.error.HTTPError as exc:
        logger.error(
            "PING FAILED — HTTP %s %s — %s%s",
            exc.code,
            exc.reason,
            args.url,
            _HEALTH_PATH,
        )
        sys.exit(1)
    except urllib.error.URLError as exc:
        logger.error(
            "PING FAILED — connection error: %s — %s%s",
            exc.reason,
            args.url,
            _HEALTH_PATH,
        )
        sys.exit(1)
    except TimeoutError:
        logger.error(
            "PING FAILED — timed out after %ss — %s%s",
            _TIMEOUT_SECONDS,
            args.url,
            _HEALTH_PATH,
        )
        sys.exit(1)

    if status_code == 200:
        logger.info("PING OK — HTTP %s — %s%s", status_code, args.url, _HEALTH_PATH)
        sys.exit(0)
    else:
        logger.error(
            "PING FAILED — HTTP %s — %s%s", status_code, args.url, _HEALTH_PATH
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
