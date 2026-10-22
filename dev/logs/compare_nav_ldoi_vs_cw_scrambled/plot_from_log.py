#!/usr/bin/env python3
"""Parse compare_nav_ldoi_vs_cw_scrambled.log and write summary figures."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
LOG_PATH = HERE / "compare_nav_ldoi_vs_cw_scrambled.log"

DONE_RE = re.compile(
    r"^scramble\s+(\d+)/\d+\s+done:\s+"
    r"cw=([\d.]+)s\s+ldoi=([\d.]+)s\s+"
    r"mean entity RMSE=([\d.]+)\s*$"
)
SCORE_RE = re.compile(
    r"^\s*flat scores:\s+"
    r"cubewalkers=([\d.]+)/([\d.]+)\s+"
    r"ldoi=([\d.]+)/([\d.]+)\s*$"
)

COLORS = ["#4C72B0", "#C44E52"]


def parse_log(path: Path) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    pending: dict[str, float | int] | None = None
    with path.open() as f:
        for line in f:
            m_done = DONE_RE.match(line)
            if m_done:
                pending = {
                    "scramble": int(m_done.group(1)),
                    "t_cubewalkers": float(m_done.group(2)),
                    "t_ldoi": float(m_done.group(3)),
                    "mean_entity_rmse": float(m_done.group(4)),
                }
                continue
            m_score = SCORE_RE.match(line)
            if m_score and pending is not None:
                pending["score_cubewalkers"] = float(m_score.group(1))
                pending["max_score"] = float(m_score.group(2))
                pending["score_ldoi"] = float(m_score.group(3))
                rows.append(pending)
                pending = None
            elif pending is not None and line.strip() == "":
                # done line without flat scores
                rows.append(pending)
                pending = None
    if pending is not None:
        rows.append(pending)
    if not rows:
        raise SystemExit(f"No scramble rows parsed from {path}")
    return pd.DataFrame(rows)


def main() -> None:
    df = parse_log(LOG_PATH)
    n = len(df)
    print(f"Parsed {n} scrambles from {LOG_PATH.name}")
    print(
        f"  mean entity RMSE={df['mean_entity_rmse'].mean():.4f}  "
        f"(median={df['mean_entity_rmse'].median():.4f})"
    )
    print(
        f"  mean times (s): cw={df['t_cubewalkers'].mean():.2f}  "
        f"ldoi={df['t_ldoi'].mean():.2f}"
    )

    # Timing
    fig, ax = plt.subplots(figsize=(5, 4))
    means_t = [df["t_cubewalkers"].mean(), df["t_ldoi"].mean()]
    ax.bar(["cubewalkers", "LDOI"], means_t, color=COLORS)
    ax.set_ylabel("Mean total wall time (s)")
    ax.set_title(f"Prediction time by method (n={n} scrambles)")
    for i, v in enumerate(means_t):
        ax.text(i, v, f"{v:.1f}s", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    out = HERE / "compare_nav_ldoi_vs_cw_scrambled_time.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Wrote {out.name}")

    # Entity RMSE distribution
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.boxplot(
        [df["mean_entity_rmse"].to_numpy()],
        tick_labels=["LDOI vs cubewalkers"],
        showfliers=True,
        widths=0.5,
    )
    jitter = np.random.default_rng(0).uniform(-0.06, 0.06, n)
    ax.scatter(
        np.ones(n) + jitter,
        df["mean_entity_rmse"],
        color=COLORS[1],
        alpha=0.7,
        s=28,
        zorder=3,
    )
    ax.set_ylabel("Mean entity RMSE vs cubewalkers")
    ax.set_title(f"Entity NAV accuracy across {n} scrambles")
    fig.tight_layout()
    out = HERE / "compare_nav_ldoi_vs_cw_scrambled_entity_rmse.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Wrote {out.name}")

    # Flat-score scatter
    if "score_cubewalkers" in df.columns and df["score_cubewalkers"].notna().any():
        fig, ax = plt.subplots(figsize=(5.5, 5))
        x = df["score_cubewalkers"].to_numpy()
        y = df["score_ldoi"].to_numpy()
        ax.scatter(x, y, color=COLORS[0], alpha=0.8, s=40, zorder=3)
        lo = float(min(x.min(), y.min()))
        hi = float(max(x.max(), y.max()))
        pad = 0.05 * (hi - lo) if hi > lo else 1.0
        ax.plot(
            [lo - pad, hi + pad],
            [lo - pad, hi + pad],
            color="gray",
            lw=0.9,
            ls="--",
            label="y = x",
        )
        if len(x) >= 2 and np.std(x) > 0 and np.std(y) > 0:
            r = float(np.corrcoef(x, y)[0, 1])
            ax.set_title(f"Flat score: cubewalkers vs LDOI  (r = {r:.3f}, n = {len(x)})")
            print(f"  flat-score r={r:+.4f}")
        else:
            ax.set_title(f"Flat score: cubewalkers vs LDOI  (n = {len(x)})")
        ax.set_xlabel("Flat score (cubewalkers)")
        ax.set_ylabel("Flat score (LDOI)")
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        out = HERE / "compare_nav_ldoi_vs_cw_scrambled_score_scatter.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"Wrote {out.name}")
    else:
        print("Skipped flat-score scatter (no scores in log)")


if __name__ == "__main__":
    main()
