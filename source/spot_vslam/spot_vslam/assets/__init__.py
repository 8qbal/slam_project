"""Spot robot configurations and USD assets used by the SLAM tasks."""

import os

SPOT_VSLAM_ASSETS_DIR = os.path.dirname(os.path.abspath(__file__))
"""Directory of this package."""

SPOT_VSLAM_USD_DIR = os.environ.get("SPOT_VSLAM_USD_DIR", os.path.join(SPOT_VSLAM_ASSETS_DIR, "usd"))
"""Directory holding the USD files (robot + maps). USD files are git-ignored, so copy them here
or point the ``SPOT_VSLAM_USD_DIR`` environment variable at them."""
