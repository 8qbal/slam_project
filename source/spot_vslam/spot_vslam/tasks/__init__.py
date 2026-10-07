"""Gym task registrations. Each task package registers its ids in ``config/<robot>/__init__.py``."""

from isaaclab_tasks.utils import import_packages

import_packages(__name__, ["utils", ".mdp", "agents"])
