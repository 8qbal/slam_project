"""Gym task registrations for Spot visual SLAM."""

from isaaclab_tasks.utils import import_packages

import_packages(__name__, ["utils", ".mdp", "agents", "params"])
