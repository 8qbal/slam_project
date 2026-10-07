"""Spot robot config for the SLAM test env: Isaac Lab's stock Spot, spawned 1 m high."""

from isaaclab_assets.robots.spot import SPOT_CFG as ISAACLAB_SPOT_CFG

SPOT_CFG = ISAACLAB_SPOT_CFG.replace(init_state=ISAACLAB_SPOT_CFG.init_state.replace(pos=(0.0, 0.0, 1.0)))
"""Isaac Lab's ``SPOT_CFG`` (same USD, actuators and joint defaults) with the base starting 1 m above the ground."""
