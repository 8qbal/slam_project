# Legacy scripts

Old experiments and test loops from the original project. Nothing in the current pipeline calls them, and they are
not kept up to date with Isaac Lab 3.0 or the task configs (some use lost USDs such as `corrider_map.usd`, or
unregistered tasks, or the old `spot_vslam.managers` / `tasks.spot_vslam` modules). Kept for reference; delete a script once nobody needs it.

- `high_level/`: early high-level training/wrapper tests; `test_environment.py` is the original map test (source of
  the Map A start/goal).
- `orb_aware/`: rule-based ORB-aware navigation.
- `slam/`: ORB-SLAM3 / Nav2 bridge and deviation tests.
- `add_light_to_spot_usd.py`: wrote a lit Spot USD next to the (lost) custom USDs.
