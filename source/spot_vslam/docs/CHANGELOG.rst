Changelog
---------

0.2.0 (2026-09-25)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Migrated the project layout to the Isaac Lab 3.0 external-project format (``pyproject.toml`` with the
  ``isaaclab.tasks`` entry point instead of ``setup.py``).
* Moved the SLAM code out of ``spot_vslam.my_project`` into ``assets``, ``envs``, ``managers``, ``mdp``,
  ``high_level`` and ``tasks.spot_vslam``; standalone scripts moved under ``scripts/``.
* Updated imports for Isaac Lab 3.0 (``isaaclab_tasks.core.velocity.mdp``,
  ``isaaclab_tasks.contrib.velocity.config.spot.mdp``, ``UniformNoiseCfg``).
* USD paths now resolve relative to ``spot_vslam/assets/usd`` instead of a hard-coded home directory.

Removed
^^^^^^^

* Template content not related to Spot SLAM: cart-double-pendulum MARL task, UI extension example,
  ``list_envs.py``, ``zero_agent.py``, ``random_agent.py``.
* Unused, non-importable copies of Isaac Lab 2.x internals (command manager, env cfg, terrain generator/importer)
  and ROS 2 term stubs that referenced non-existent Isaac Lab classes.

0.1.0 (2025-08-11)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Created an initial template for building an extension or project based on Isaac Lab
