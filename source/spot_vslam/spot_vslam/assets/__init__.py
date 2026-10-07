"""Robot and map assets used by the Spot SLAM tasks.

USD files come from the local Isaac Sim asset pack (``ISAACSIM_ASSET_ROOT``), never from this repo.
"""

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

SPOT_USD_PATH = f"{ISAAC_NUCLEUS_DIR}/Robots/BostonDynamics/spot/spot.usd"
"""Stock Isaac Sim Spot. Cameras are attached via ``CameraCfg``."""

WAREHOUSE_USD_PATH = f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse.usd"
"""Stock Isaac Sim Simple_Warehouse (empty): the paper's Map A ("Room")."""

WAREHOUSE_SHELVES_USD_PATH = f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd"
"""Simple_Warehouse with three shelf columns: the paper's Map B ("Warehouse")."""

from .maps import PAPER_MAP_SPACING, PAPER_MAPS, PaperMap  # noqa: E402
