"""
Constants module for bouldering analysis application.

This module contains shared constants used across the application,
including hold type mappings.
"""

from typing import Dict

# Hold type mapping from model class IDs to hold type names
HOLD_TYPES: Dict[int, str] = {
    0: "crimp",
    1: "jug",
    2: "sloper",
    3: "pinch",
    4: "pocket",
    5: "foot-hold",
    6: "start-hold",
    7: "top-out-hold",
}
