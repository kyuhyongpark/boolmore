# %% [markdown]
# RA 243 — how many nodes does LDOI fix under random source combos?
#
# On ``case_study/RA/243`` primes:
#
# 1. Load / pickle primes; count syntactic sources
# 2. Sample many full 0/1 source assignments (seeded; default 2000)
# 3. For each combo: LDOI → ``n_fixed`` / ``n_free``
# 4. Persist per-combo table + summary plots under ``dev/logs/``
#
# ``SMOKE=1`` uses a tiny sample for an end-to-end check.
# Sibling of ``investigate_ldoi_ra_243_source_combos.py``.

# %%
from __future__ import annotations

import json
import os
import pickle
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from boolmore.inference.ldoi import logical_domain_of_influence
from boolmore.io.load import load_primes
from boolmore.io.paths import project_root

_DEV = Path(__file__).resolve().parent
if str(_DEV) not in sys.path:
    sys.path.insert(0, str(_DEV))

from nav_helpers import (
    Tee,
    decode_source_index,
    sample_source_indices,
    source_nodes_from_primes,
)

# %% [markdown]
# ## Config

# %%
ROOT = project_root()
CASE = ROOT / "case_study" / "RA"
MODEL_STEM = CASE / "243"
BNET = MODEL_STEM.with_suffix(".bnet")
PKL = MODEL_STEM.with_suffix(".pkl")

LOGS_ROOT = Path(__file__).resolve().parent / "logs"
SCRIPT_NAME = Path(__file__).stem

COMBO_RNG_SEED = 0
N_COMBOS = 10000
PROGRESS_EVERY = 100

FORCE_RECOMPUTE = False

SMOKE = os.environ.get("SMOKE") == "1"
OUT_TAG = "_smoke" if SMOKE else ""
if SMOKE:
    N_COMBOS = 40
    PROGRESS_EVERY = 10

# %% [markdown]
# ## Load primes + sample combos

# %%
RUN_DIR = (
    LOGS_ROOT
    / SCRIPT_NAME
    / MODEL_STEM.name
    / f"combo-seed-{COMBO_RNG_SEED}_n{N_COMBOS}{OUT_TAG}"
)
RUN_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = (
    RUN_DIR
    / f"{SCRIPT_NAME}_{MODEL_STEM.name}_combo-seed-{COMBO_RNG_SEED}"
    f"_n{N_COMBOS}{OUT_TAG}.log"
)
_log_file = LOG_PATH.open("w", encoding="utf-8")
_stdout = sys.stdout
sys.stdout = Tee(_stdout, _log_file)

print(f"run dir: {RUN_DIR}")
print(f"log file: {LOG_PATH}")
print(f"started (UTC): {datetime.now(timezone.utc).isoformat()}")
print(f"model bnet: {BNET}")
print(f"model pkl:  {PKL}")
print(f"COMBO_RNG_SEED={COMBO_RNG_SEED}  N_COMBOS={N_COMBOS}")
print(f"FORCE_RECOMPUTE={FORCE_RECOMPUTE}  SMOKE={SMOKE}")
print()

if PKL.is_file() and not FORCE_RECOMPUTE:
    print(f"loading primes from {PKL}")
    primes = load_primes(PKL)
else:
    if not BNET.is_file():
        raise SystemExit(f"missing bnet: {BNET}")
    print(f"importing primes from {BNET}…")
    _t0 = time.perf_counter()
    primes = load_primes(BNET)
    print(f"  load_primes done in {time.perf_counter() - _t0:.3f}s")
    with PKL.open("wb") as f:
        pickle.dump(primes, f)
    print(f"wrote primes → {PKL}")

node_names = sorted(primes)
n_nodes = len(primes)
sources = source_nodes_from_primes(primes)
n_sources = len(sources)
if n_sources == 0:
    raise SystemExit("no syntactic source nodes found after import")

print(f"Model nodes : {n_nodes}")
print(f"Source nodes: {n_sources}")
print(f"  first sources: {', '.join(sources[:8])}{'…' if n_sources > 8 else ''}")
print()

COMBOS_PATH = RUN_DIR / "combos.json"
if COMBOS_PATH.is_file() and not FORCE_RECOMPUTE:
    combos_payload = json.loads(COMBOS_PATH.read_text(encoding="utf-8"))
    combo_indices = [int(x) for x in combos_payload["indices"]]
    print(f"loaded {len(combo_indices)} combo indices from {COMBOS_PATH.name}")
else:
    rng = random.Random(COMBO_RNG_SEED)
    combo_indices = sample_source_indices(n_sources, N_COMBOS, rng)
    combos_payload = {
        "combo_rng_seed": COMBO_RNG_SEED,
        "n_sources": n_sources,
        "sources": list(sources),
        "n_combos_requested": N_COMBOS,
        "indices": combo_indices,
        "smoke": SMOKE,
    }
    COMBOS_PATH.write_text(
        json.dumps(combos_payload, indent=2) + "\n", encoding="utf-8"
    )
    print(f"sampled {len(combo_indices)} combo indices → {COMBOS_PATH}")

combo_indices = combo_indices[:N_COMBOS]
print(f"using {len(combo_indices)} source assignments")
print()

# %% [markdown]
# ## LDOI fixed-node counts (cached)

# %%
CACHE_PATH = RUN_DIR / f"ldoi_fixed_counts{OUT_TAG}.pkl"
cache: dict = {}
if CACHE_PATH.is_file() and not FORCE_RECOMPUTE:
    with CACHE_PATH.open("rb") as f:
        cache = pickle.load(f)
    print(
        f"loaded fixed-count cache "
        f"({len(cache.get('rows', []))} rows) from {CACHE_PATH.name}"
    )

rows: list[dict] = [] if FORCE_RECOMPUTE else list(cache.get("rows", []))
done_keys: set[str] = {
    json.dumps(sorted(r["assignment"].items()), separators=(",", ":"))
    for r in rows
    if "assignment" in r
}

_t_all = time.perf_counter()
for i, idx in enumerate(combo_indices):
    assignment = decode_source_index(idx, sources, {})
    key = json.dumps(sorted(assignment.items()), separators=(",", ":"))
    if key in done_keys and not FORCE_RECOMPUTE:
        if (i + 1) % PROGRESS_EVERY == 0 or i == 0:
            print(f"[{i + 1}/{len(combo_indices)}] cache hit  index={idx}")
        continue

    _t0 = time.perf_counter()
    fixed = logical_domain_of_influence(primes, assignment)
    dt = time.perf_counter() - _t0
    n_fixed = len(fixed)
    n_free = n_nodes - n_fixed
    row = {
        "combo_i": i,
        "combo_index": int(idx) if idx.bit_length() < 63 else str(idx),
        "n_sources_on": int(sum(assignment.values())),
        "n_fixed": int(n_fixed),
        "n_free": int(n_free),
        "frac_fixed": float(n_fixed / n_nodes),
        "frac_free": float(n_free / n_nodes),
        "ldoi_s": float(dt),
        "assignment": dict(assignment),
    }
    rows.append(row)
    done_keys.add(key)

    if (i + 1) % PROGRESS_EVERY == 0 or i == 0 or i + 1 == len(combo_indices):
        print(
            f"[{i + 1}/{len(combo_indices)}] index={idx}  "
            f"n_fixed={n_fixed}  n_free={n_free}  "
            f"frac_fixed={n_fixed / n_nodes:.3f}  ldoi={dt:.4f}s"
        )

    if (i + 1) % PROGRESS_EVERY == 0 or i + 1 == len(combo_indices):
        with CACHE_PATH.open("wb") as f:
            pickle.dump(
                {
                    "model_pkl": str(PKL),
                    "n_nodes": n_nodes,
                    "n_sources": n_sources,
                    "sources": list(sources),
                    "combo_rng_seed": COMBO_RNG_SEED,
                    "n_combos": len(combo_indices),
                    "smoke": SMOKE,
                    "rows": rows,
                },
                f,
            )

t_total = time.perf_counter() - _t_all
print(f"LDOI pass done in {t_total:.3f}s  cache → {CACHE_PATH}")
print()

# Align rows to current combo order for a tidy table (drop stale extras).
by_key = {
    json.dumps(sorted(r["assignment"].items()), separators=(",", ":")): r
    for r in rows
}
ordered_rows: list[dict] = []
for i, idx in enumerate(combo_indices):
    assignment = decode_source_index(idx, sources, {})
    key = json.dumps(sorted(assignment.items()), separators=(",", ":"))
    r = dict(by_key[key])
    r["combo_i"] = i
    ordered_rows.append(r)

df = pd.DataFrame(
    [
        {
            "combo_i": r["combo_i"],
            "combo_index": r["combo_index"],
            "n_sources_on": r["n_sources_on"],
            "n_fixed": r["n_fixed"],
            "n_free": r["n_free"],
            "frac_fixed": r["frac_fixed"],
            "frac_free": r["frac_free"],
            "ldoi_s": r["ldoi_s"],
        }
        for r in ordered_rows
    ]
)

print("Fixed / free summary across combos:")
print(
    df[["n_fixed", "n_free", "frac_fixed", "frac_free", "ldoi_s"]]
    .describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
    .to_string(float_format=lambda x: f"{x:.4f}")
)
print()
print(
    f"n_fixed:  mean={df['n_fixed'].mean():.1f}  "
    f"median={df['n_fixed'].median():.1f}  "
    f"min={df['n_fixed'].min()}  max={df['n_fixed'].max()}"
)
print(
    f"n_free:   mean={df['n_free'].mean():.1f}  "
    f"median={df['n_free'].median():.1f}  "
    f"min={df['n_free'].min()}  max={df['n_free'].max()}"
)
print(
    f"fully fixed combos (n_free=0): "
    f"{int((df['n_free'] == 0).sum())} / {len(df)}"
)
print()

# %% [markdown]
# ## Plots + persist


# %%
def _plot_hist(series: np.ndarray, *, xlabel: str, title: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.hist(series, bins=40, color="#1f77b4", edgecolor="white", alpha=0.85)
    ax.axvline(
        float(np.mean(series)),
        color="#d62728",
        linestyle="--",
        linewidth=1.5,
        label=f"mean={np.mean(series):.1f}",
    )
    ax.axvline(
        float(np.median(series)),
        color="#2ca02c",
        linestyle=":",
        linewidth=1.5,
        label=f"median={np.median(series):.1f}",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Combos")
    ax.set_title(title)
    xmax = float(np.nanmax(series)) if len(series) else 0.0
    ax.set_xlim(0.0, xmax * 1.02 if xmax > 0 else 1.0)
    ax.set_ylim(bottom=0.0)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _plot_ecdf(
    series: np.ndarray,
    *,
    xlabel: str,
    title: str,
    path: Path,
    xlim: tuple[float, float] | None = None,
) -> Path:
    vals = np.sort(series.astype(np.float64))
    frac = np.arange(1, len(vals) + 1) / len(vals)
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.step(vals, frac, where="post", color="#1f77b4", linewidth=2.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("ECDF  (fraction of combos ≤ x)")
    ax.set_title(title)
    if xlim is not None:
        ax.set_xlim(*xlim)
    else:
        xmax = float(vals[-1]) if len(vals) else 0.0
        ax.set_xlim(0.0, xmax * 1.02 if xmax > 0 else 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _plot_scatter_sources_vs_fixed(path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.scatter(
        df["n_sources_on"],
        df["n_fixed"],
        s=12,
        alpha=0.35,
        color="#1f77b4",
        edgecolors="none",
    )
    ax.set_xlabel("Sources ON in assignment")
    ax.set_ylabel("Nodes fixed by logical percolation")
    ax.set_title("Logical percolation fixed count vs source ON-count")
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _plot_box_summary(path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
    for ax, col, label in zip(
        axes,
        ["n_fixed", "n_free"],
        ["Nodes fixed by logical percolation", "Nodes still free"],
    ):
        bp = ax.boxplot(
            [df[col].to_numpy()],
            tick_labels=[col],
            showfliers=True,
            whis=(5, 95),
            widths=0.45,
            patch_artist=True,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor("#1f77b4")
            patch.set_alpha(0.45)
        ax.scatter(
            [1],
            [df[col].mean()],
            color="#d62728",
            marker="D",
            s=40,
            zorder=4,
            label="mean",
        )
        ax.set_ylabel(label)
        ax.set_ylim(bottom=0.0)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(frameon=False, loc="best")
    fig.suptitle(
        f"RA 243 logical percolation fix counts  (n_combos={len(df)}, N={n_nodes})",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


print("Writing plots…")
plot_paths = [
    _plot_hist(
        df["n_fixed"].to_numpy(),
        xlabel="n_fixed (logical percolation)",
        title=f"RA 243: logical percolation fixed-node count  (n={len(df)} combos)",
        path=RUN_DIR / "n_fixed_hist.png",
    ),
    _plot_hist(
        df["n_free"].to_numpy(),
        xlabel="n_free (N − n_fixed)",
        title=(
            f"RA 243: free-node count after logical percolation  "
            f"(n={len(df)} combos)"
        ),
        path=RUN_DIR / "n_free_hist.png",
    ),
    _plot_ecdf(
        df["n_fixed"].to_numpy(),
        xlabel="n_fixed (logical percolation)",
        title="RA 243: ECDF of logical percolation fixed-node count",
        path=RUN_DIR / "n_fixed_ecdf.png",
    ),
    _plot_ecdf(
        df["frac_fixed"].to_numpy(),
        xlabel="fraction of nodes fixed",
        title="RA 243: ECDF of logical percolation fixed fraction",
        path=RUN_DIR / "frac_fixed_ecdf.png",
        xlim=(0.0, 1.0),
    ),
    _plot_ecdf(
        df["n_free"].to_numpy(),
        xlabel="n_free (N − n_fixed)",
        title="RA 243: ECDF of free-node count after logical percolation",
        path=RUN_DIR / "n_free_ecdf.png",
    ),
    _plot_ecdf(
        df["frac_free"].to_numpy(),
        xlabel="fraction of nodes free",
        title="RA 243: ECDF of free-node fraction after logical percolation",
        path=RUN_DIR / "frac_free_ecdf.png",
        xlim=(0.0, 1.0),
    ),
    _plot_scatter_sources_vs_fixed(RUN_DIR / "n_sources_on_vs_n_fixed.png"),
    _plot_box_summary(RUN_DIR / "n_fixed_n_free_box.png"),
]
for p in plot_paths:
    print(f"  wrote {p}")

csv_path = RUN_DIR / "ldoi_fixed_counts.csv"
summary_path = RUN_DIR / "ldoi_fixed_summary.json"
df.to_csv(csv_path, index=False)

summary = {
    "model_pkl": str(PKL),
    "n_nodes": n_nodes,
    "n_sources": n_sources,
    "n_combos": int(len(df)),
    "combo_rng_seed": COMBO_RNG_SEED,
    "smoke": SMOKE,
    "n_fixed": {
        "mean": float(df["n_fixed"].mean()),
        "median": float(df["n_fixed"].median()),
        "std": float(df["n_fixed"].std(ddof=0)),
        "min": int(df["n_fixed"].min()),
        "max": int(df["n_fixed"].max()),
        "p05": float(df["n_fixed"].quantile(0.05)),
        "p95": float(df["n_fixed"].quantile(0.95)),
    },
    "n_free": {
        "mean": float(df["n_free"].mean()),
        "median": float(df["n_free"].median()),
        "std": float(df["n_free"].std(ddof=0)),
        "min": int(df["n_free"].min()),
        "max": int(df["n_free"].max()),
        "p05": float(df["n_free"].quantile(0.05)),
        "p95": float(df["n_free"].quantile(0.95)),
    },
    "n_fully_fixed_combos": int((df["n_free"] == 0).sum()),
    "ldoi_s": {
        "total": float(df["ldoi_s"].sum()),
        "mean": float(df["ldoi_s"].mean()),
    },
    "wall_s_pass": float(t_total),
}
summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(f"  wrote {csv_path}")
print(f"  wrote {summary_path}")

print()
print(f"finished (UTC): {datetime.now(timezone.utc).isoformat()}")

# %%
sys.stdout = _stdout
_log_file.close()
print(f"log written: {LOG_PATH}")
print(f"outputs under: {RUN_DIR}")
