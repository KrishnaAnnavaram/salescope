"""salescope: honest cross-sectional sales prediction for BigMart-style item-outlet data.

The package treats BigMart as tabular regression (it has no dates), evaluates every model under
random, new-outlet and new-item cross-validation, and keeps forecasting in a separate track that
only accepts data with real timestamps.
"""

__version__ = "0.1.0"

from salescope.schema import ID_COL, OUTLET_COL, TARGET  # noqa: E402

__all__ = ["ID_COL", "OUTLET_COL", "TARGET", "__version__"]
