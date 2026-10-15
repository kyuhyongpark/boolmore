# %% [markdown]
# Unfixed source nodes → lower bound on minimal trap spaces
#
# Load the breast-cancer trial NAV data and the parsimonious model (`.pkl` / `.bnet`).
# For each distinct intervention (`fixes`), count how many **source** nodes of the
# model are fixed vs left free. With `x` unfixed sources, a lower bound on the
# number of minimal trap spaces under that intervention is `2**x` (one source
# assignment per free binary input).

# %%
from __future__ import annotations

from collections import Counter

import pandas as pd

from boolmore.io.load import import_NAV_exps, load_primes
from boolmore.io.paths import project_root

# %%
ROOT = project_root()
CASE = ROOT / "case_study" / "breast_cancer"

DATA = CASE / "breast_cancer_data_trial.csv"
PKL = CASE / "BreastCancer_primes_parsimonious.pkl"
BNET = CASE / "BreastCancer_merged_parsimonious.bnet"

# Same column roles as evaluate_model.py / BreastCancer_config.json
ID_COL = "id"
WEIGHT_COL = "Weight"
OBSERVED_COL = "Observed node"
OUTCOME_COL = "Value"
FIXES_COLS = (
    "Signals",
    "Drug treatment/intervention",
    "Cell line characteristic profile",
    "Additional features of the cell population",
)

# %% [markdown]
# ## Load model and data
#
# Source nodes are pure inputs: `primes[node] == [[{node: 0}], [{node: 1}]]`.

# %%
primes_pkl = load_primes(PKL)
primes_bnet = load_primes(BNET)

assert primes_pkl == primes_bnet, "pkl and bnet primes differ"
primes = primes_pkl

source_nodes = sorted(
    node for node in primes if primes[node] == [[{node: 0}], [{node: 1}]]
)
source_set = set(source_nodes)

print(f"Model nodes : {len(primes)}")
print(f"Source nodes: {len(source_nodes)}")
print(source_nodes)

# %%
exps = import_NAV_exps(
    DATA,
    id_col=ID_COL,
    weight_col=WEIGHT_COL,
    observed_col=OBSERVED_COL,
    outcome_col=OUTCOME_COL,
    fixes_cols=FIXES_COLS,
)

unique_fixes = sorted({exp.fixes for exp in exps}, key=lambda f: (len(f), f))
print(f"Experiments     : {len(exps)}")
print(f"Distinct fixes  : {len(unique_fixes)}")

# %% [markdown]
# ## Per-fix source coverage and `2**x` lower bound

# %%
def analyze_fixes(fixes, source_set: set[str]) -> dict:
    fixed_nodes = {node for node, _ in fixes}
    fixed_sources = sorted(fixed_nodes & source_set)
    unfixed_sources = sorted(source_set - fixed_nodes)
    x = len(unfixed_sources)
    return {
        "n_fixes": len(fixes),
        "n_fixed_sources": len(fixed_sources),
        "n_unfixed_sources": x,
        "min_trap_lower_bound": 2**x,
        "fixed_sources": fixed_sources,
        "unfixed_sources": unfixed_sources,
        "fixes": fixes,
    }


rows = [analyze_fixes(fixes, source_set) for fixes in unique_fixes]
df = pd.DataFrame(rows)

# Attach experiment ids that share each intervention
fixes_to_ids: dict = {}
for exp in exps:
    fixes_to_ids.setdefault(exp.fixes, []).append(exp.id)
df["exp_ids"] = [fixes_to_ids[f] for f in df["fixes"]]
df["n_exps"] = df["exp_ids"].map(len)

display_cols = [
    "n_exps",
    "n_fixes",
    "n_fixed_sources",
    "n_unfixed_sources",
    "min_trap_lower_bound",
]
summary = df[display_cols].sort_values(
    ["n_unfixed_sources", "n_exps"], ascending=[False, False]
).reset_index(drop=True)
summary

# %%
hist = Counter(df["n_unfixed_sources"])
hist_df = pd.DataFrame(
    [
        {
            "n_unfixed_sources (x)": x,
            "n_distinct_fixes": hist[x],
            "min_trap_lower_bound (2**x)": 2**x,
        }
        for x in sorted(hist)
    ]
)
print(
    f"Source nodes in model: {len(source_nodes)}\n"
    f"Unfixed sources — min {df['n_unfixed_sources'].min()}, "
    f"median {df['n_unfixed_sources'].median():.0f}, "
    f"max {df['n_unfixed_sources'].max()}"
)
hist_df

# %% [markdown]
# ## Worst / best cases (most / fewest unfixed sources)

# %%
def show_case(row: pd.Series, title: str) -> None:
    print(title)
    print(f"  exp ids              : {row['exp_ids'][:12]}{'...' if len(row['exp_ids']) > 12 else ''}")
    print(f"  n_exps               : {row['n_exps']}")
    print(f"  total fixed nodes    : {row['n_fixes']}")
    print(f"  fixed sources        : {row['n_fixed_sources']}  {row['fixed_sources']}")
    print(f"  unfixed sources (x)  : {row['n_unfixed_sources']}")
    print(f"  lower bound 2**x     : {row['min_trap_lower_bound']:,}")
    print(f"  unfixed sources      : {row['unfixed_sources']}")
    print(f"  fixes                : {row['fixes']}")
    print()


worst = df.loc[df["n_unfixed_sources"].idxmax()]
best = df.loc[df["n_unfixed_sources"].idxmin()]
show_case(worst, "Most unfixed sources (largest lower bound)")
show_case(best, "Fewest unfixed sources (smallest lower bound)")

# %%
# Full detail table (sorted by descending lower bound)
detail = df.sort_values(
    ["n_unfixed_sources", "n_exps"], ascending=[False, False]
).reset_index(drop=True)
detail[
    [
        "n_exps",
        "exp_ids",
        "n_fixes",
        "n_fixed_sources",
        "n_unfixed_sources",
        "min_trap_lower_bound",
        "fixed_sources",
        "unfixed_sources",
        "fixes",
    ]
]
