"""Paper Tables 7-8 (system load) from the performance logs of load_high_level.py runs.

Expects <root>/<config>/seed<N>/performance_log_*_final.csv (the layout run_system_load_test.sh writes) and prints
mean +- std over the seeds of each config. Pure pandas, no Isaac Sim needed.

    python scripts/high_level/summarize_system_load.py output/system_load_test
"""

import argparse
import glob
import os

import pandas as pd

CONFIGS = [("rl_only", "RL only"), ("rl_vslam", "RL + VSLAM"), ("rl_vslam_tsdf", "RL + VSLAM + TSDF")]


def per_run(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)
    return {
        "Avg FPS": df.System_FPS.mean(),
        "Min FPS": df.System_FPS.min(),
        "RAM start (GB)": df.RAM_Usage_GB.iloc[0],
        "RAM end (GB)": df.RAM_Usage_GB.iloc[-1],
        "Avg CPU (%)": df.CPU_Usage_Percent.mean(),
        "Peak CPU (%)": df.CPU_Usage_Percent.max(),
        "Avg read (MB/s)": df.Disk_Read_MBs.mean(),
        "Avg write (MB/s)": df.Disk_Write_MBs.mean(),
        "Peak write (MB/s)": df.Disk_Write_MBs.max(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", help="output folder of run_system_load_test.sh")
    root = parser.parse_args().root

    rows = []
    for cfg, _ in CONFIGS:
        for csv_path in sorted(glob.glob(os.path.join(root, cfg, "seed*", "performance_log_*_final.csv"))):
            seed = os.path.basename(os.path.dirname(csv_path)).removeprefix("seed")
            rows.append({"config": cfg, "seed": seed, **per_run(csv_path)})
    if not rows:
        raise SystemExit(f"No performance_log_*_final.csv found under {root}/<config>/seed*/")
    runs = pd.DataFrame(rows)
    runs.to_csv(os.path.join(root, "all_trials.csv"), index=False)

    metrics = [c for c in runs.columns if c not in ("config", "seed")]
    table = {}
    for cfg, name in CONFIGS:
        g = runs[runs.config == cfg]
        if len(g):
            table[f"{name} (n={len(g)})"] = {m: f"{g[m].mean():.2f} ± {g[m].std(ddof=1):.2f}" for m in metrics}
    table = pd.DataFrame(table)
    print("Table 7 (system load) and Table 8 (disk I/O), mean ± std over runs:")
    print(table.to_string())
    print(f"\nPer-run values: {os.path.join(root, 'all_trials.csv')}")


if __name__ == "__main__":
    main()
