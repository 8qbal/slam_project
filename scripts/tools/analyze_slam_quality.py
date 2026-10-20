"""Do cuVSLAM's quality signals tell when its pose is wrong? Reads ``--trace_npz`` files of eval_nav_protocol.py.

For every step of every run: the signal at that step (and its mean over the last 10 steps) against two labels,
"the error is above E metres now" and "the error will grow by more than G metres within 25 steps". The score is the
AUROC (0.5 = no information, 1 = perfect; below 0.5 = the signal goes the other way).

    uv run --no-sync python scripts/tools/analyze_slam_quality.py output/slam_quality/trace_map*.npz
"""

import sys

import numpy as np

ERR_HIGH, GROW, AHEAD = 1.5, 0.5, 25


def auroc(score, label):
    """P(score of a positive > score of a negative), ties count half (Mann-Whitney U)."""
    pos, neg = score[label], score[~label]
    if len(pos) < 20 or len(neg) < 20:
        return float("nan")
    r = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1.0
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def rolling_mean(x, k=10):
    c = np.cumsum(np.pad(x, ((k, 0), (0, 0))), axis=0)
    return (c[k:] - c[:-k]) / k


rows = {k: [] for k in ("features", "features10", "odom_slam", "track_ms", "err", "future")}
for path in sys.argv[1:]:
    d = np.load(path)
    err, live = d["err"], d["live"] > 0.5
    gap = np.nan_to_num(d["odom_slam"], nan=0.0)  # no loop-closure pose yet: no gap
    future = np.full_like(err, np.nan)
    future[:-AHEAD] = err[AHEAD:] - err[:-AHEAD]
    keep = live & np.isfinite(future)
    signals = {
        "features": d["features"],
        "features10": rolling_mean(d["features"]),
        "odom_slam": gap,
        "track_ms": d["track_ms"],
        "err": err,
        "future": future,
    }
    for k, v in signals.items():
        rows[k].append(v[keep])
    share = (err[live] > ERR_HIGH).mean()
    print(f"{path}: {int(live.sum())} steps, mean error {err[live].mean():.2f} m, above {ERR_HIGH} m: {share:.2f}")
cat = {k: np.concatenate(v) for k, v in rows.items()}
now, soon = cat["err"] > ERR_HIGH, cat["future"] > GROW
print(
    f"\nsteps {len(now)}; error above {ERR_HIGH} m now: {now.mean():.2f}; "
    f"grows > {GROW} m within {AHEAD} steps: {soon.mean():.2f}"
)
print("signal        AUROC(err now)  AUROC(err grows)   [far from 0.5 = informative]")
for k in ("features", "features10", "odom_slam", "track_ms"):
    print(f"{k:12s}  {auroc(cat[k], now):14.2f}  {auroc(cat[k], soon):16.2f}")
print("status        1 in every step (no information)")
