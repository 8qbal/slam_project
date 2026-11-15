"""Goal-distance curriculum for the random-tile training tasks.

Goals 6-18 m away in a straight line were too much from the start (a first run stayed at about 20 % success, all of it
short goals). The farthest allowed goal distance starts at ``START_MAX`` and moves in steps of ``STEP`` m: up when the
robot reached the goal in at least ``UP`` of the last ``WINDOW`` episodes, down when it did in less than ``DOWN``.
The nearest allowed distance follows it, so at the end the range is the plain (6, 18) m of the evaluation.

``goal_distance_curriculum`` is a curriculum term (it runs in ``_reset_idx`` right before the reset events, with the
termination flags of the episodes that just ended); ``reset_in_maze`` reads the range from ``env._goal_curriculum``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["aisle_success_rate", "goal_distance_curriculum", "goal_range", "next_goal_max"]

START_MAX = 6.0
FINAL_MAX = 18.0
STEP = 1.0
WINDOW = 48
UP = 0.6
DOWN = 0.3
MIN_LO = 2.0
FINAL_LO = 6.0


def goal_range(goal_max: float) -> tuple[float, float]:
    """(nearest, farthest) goal distance in metres for a farthest distance ``goal_max``."""
    return min(FINAL_LO, max(MIN_LO, goal_max - 4.0)), goal_max


def next_goal_max(goal_max: float, wins: int, total: int) -> float:
    """New farthest goal distance after ``total`` episodes of which ``wins`` reached the goal."""
    rate = wins / max(total, 1)
    if rate >= UP:
        return min(goal_max + STEP, FINAL_MAX)
    if rate < DOWN:
        return max(goal_max - STEP, START_MAX)
    return goal_max


def goal_distance_curriculum(env: ManagerBasedRLEnv, env_ids) -> float:
    state = getattr(env, "_goal_curriculum", None)
    if state is None:
        state = env._goal_curriculum = {"max": START_MAX, "wins": 0, "total": 0}
    aisle = getattr(env, "_aisle_ep", None)  # the evaluation-like episodes (reset_in_maze aisle_prob) do not count here
    ids = env_ids if aisle is None else env_ids[~aisle[env_ids]]
    if env.common_step_counter > 0 and len(ids) > 0:  # the first reset has no finished episodes
        state["wins"] += int(env.termination_manager.get_term("goal_reached")[ids].sum())
        state["total"] += len(ids)
        if state["total"] >= WINDOW:
            state["max"] = next_goal_max(state["max"], state["wins"], state["total"])
            state["wins"] = state["total"] = 0
    return state["max"]


AISLE_WINDOW = 96


def aisle_success_rate(env: ManagerBasedRLEnv, env_ids) -> float:
    """Share of the recent aisle episodes (start in the front cross aisle, goal deep in an aisle) that reached the goal;
    only logged (Curriculum/aisle_rate), it changes nothing."""
    state = getattr(env, "_aisle_stats", None)
    if state is None:
        state = env._aisle_stats = {"wins": 0.0, "total": 0.0}
    aisle = getattr(env, "_aisle_ep", None)
    if aisle is not None and env.common_step_counter > 0 and len(env_ids) > 0:
        ids = env_ids[aisle[env_ids]]
        state["wins"] += float(env.termination_manager.get_term("goal_reached")[ids].sum())
        state["total"] += float(len(ids))
        if state["total"] > AISLE_WINDOW:  # exponential window: halve both counts
            state["wins"] *= 0.5
            state["total"] *= 0.5
    return state["wins"] / max(state["total"], 1.0)
