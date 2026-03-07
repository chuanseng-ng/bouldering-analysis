"""Project-wide domain constants shared across multiple modules.

These are stable values that appear in more than one module.  Each constant
is defined here once and imported wherever it is needed.  Keeping them in a
single place prevents silent divergence when limits are tuned.
"""

from typing import Final

# Maximum number of holds accepted by any function that processes a hold list.
# At n=500, graph edge construction performs ~125,000 comparisons (O(n²)) and
# feature extraction iterates three times — both well within a single API
# request budget.  Raise this constant here to relax the cap everywhere.
MAX_HOLD_COUNT: Final[int] = 500
