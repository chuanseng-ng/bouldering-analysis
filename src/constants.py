"""
Constants module for bouldering analysis application.

This module serves as the single source of truth for shared constants
across the application, preventing duplication and ensuring consistency.
"""

# Hold types definition - single source of truth
# Structure: list of tuples (id, name, description)
HOLD_TYPES = [
    (0, "crimp", "Small, narrow hold requiring crimping fingers"),
    (1, "jug", "Large, easy-to-hold jug"),
    (2, "sloper", "Round, sloping hold that requires open-handed grip"),
    (3, "pinch", "Hold that requires pinching between thumb and fingers"),
    (4, "pocket", "Small hole that fingers fit into"),
    (5, "foot-hold", "Hold specifically for feet"),
    (6, "start-hold", "Starting hold for the route"),
    (7, "top-out-hold", "Hold used to complete the route"),
]
