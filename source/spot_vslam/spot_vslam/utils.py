"""Small helpers shared by the package and the scripts."""

import math

import torch


def yaw_from_quat_xyzw(q):
    """Yaw of an (x, y, z, w) quaternion (Isaac Lab 3.0 order).

    A tensor of shape (..., 4) gives a tensor of shape (...); four numbers (list, tuple, numpy array) give a float.
    """
    if isinstance(q, torch.Tensor):
        x, y, z, w = q.unbind(dim=-1)
        return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    x, y, z, w = (float(v) for v in q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def step_obs_info(env, action):
    """``env.step`` returning only (obs, info), for both the 4-tuple (gym) and 5-tuple (gymnasium) step APIs."""
    ret = env.step(action)
    if not isinstance(ret, tuple):
        return ret, {}
    if len(ret) in (4, 5):
        return ret[0], ret[-1]
    return ret[0], {}
