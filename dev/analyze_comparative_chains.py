# %% [markdown]
# Comparative lower/higher chains (with entity identity)
#
# Entity = `(fixes, observed)`. Comparative `lower`/`higher` edges between
# ids are lifted to entities. Report connected-group **size**, undirected
# shape (path vs branched), and directed **depth** (entities on the longest
# ascending lower→higher path: `A < B` → 2, `A < B < C` → 3).

# %%
from __future__ import annotations

from collections import Counter, defaultdict

import networkx as nx
import pandas as pd

from boolmore.io.load import import_NAV_exps
from boolmore.io.paths import project_root
from boolmore.outcome import ComparativeOutcome

# %%
DATA = project_root() / "case_study" / "breast_cancer" / "breast_cancer_data_trial.csv"
FIXES_COLS = (
    "Signals",
    "Drug treatment/intervention",
    "Cell line characteristic profile",
    "Additional features of the cell population",
)

exps = import_NAV_exps(
    DATA,
    id_col="id",
    weight_col="Weight",
    observed_col="Observed node",
    outcome_col="Value",
    fixes_cols=FIXES_COLS,
)
by_id = {e.id: e for e in exps}

# %% [markdown]
# ## Build entity graph
#
# Undirected `G` for connectivity; directed `D` (lower → higher) for depth.

# %%
def entity_of(exp):
    return (exp.fixes, exp.observed)


def label(ent) -> str:
    return f"{ent[1]}:{sorted(entity_to_ids[ent])}"


entity_to_ids: dict[tuple, set[int]] = defaultdict(set)
for e in exps:
    entity_to_ids[entity_of(e)].add(e.id)

G, D = nx.Graph(), nx.DiGraph()
id_pairs: set[frozenset[int]] = set()
self_pairs: set[frozenset[int]] = set()

for e in exps:
    if not (
        isinstance(e.outcome, ComparativeOutcome)
        and e.outcome.relation in ("higher", "lower")
    ):
        continue
    other = by_id.get(e.outcome.other_id)
    if other is None:
        continue
    id_pairs.add(frozenset((e.id, other.id)))
    ea, eb = entity_of(e), entity_of(other)
    if ea == eb:
        self_pairs.add(frozenset((e.id, other.id)))
        continue
    G.add_edge(ea, eb)
    if e.outcome.relation == "lower":
        D.add_edge(ea, eb)
    else:
        D.add_edge(eb, ea)

n_multi = sum(1 for ids in entity_to_ids.values() if len(ids) > 1)
print(f"id pairs {len(id_pairs)}  →  entity edges {G.number_of_edges()}")
print(f"identity merges (≥2 ids): {n_multi}  |  self-comparisons: {len(self_pairs)}")
if self_pairs:
    for pair in self_pairs:
        a, b = sorted(pair)
        print(f"  {a}↔{b}  {entity_of(by_id[a])[1]}")

# %% [markdown]
# ## Classify connected groups
#
# Path = undirected degrees `[1,1,2,…,2]`. Depth = `#entities` on
# `nx.dag_longest_path` in `D`.

# %%
def is_path(H: nx.Graph) -> bool:
    n = H.number_of_nodes()
    if n <= 1:
        return True
    return sorted(d for _, d in H.degree()) == [1, 1] + [2] * (n - 2)


rows = []
for cc in nx.connected_components(G):
    H, HD = G.subgraph(cc), D.subgraph(cc)
    ascent = nx.dag_longest_path(HD) if nx.is_directed_acyclic_graph(HD) else []
    rows.append(
        {
            "size": len(cc),
            "depth": len(ascent) or None,
            "path": is_path(H),
            "observed": sorted({e[1] for e in cc}),
            "ascent": ascent,
            "H": H,
        }
    )

groups = pd.DataFrame(rows).sort_values(
    ["depth", "size"], ascending=[True, False], ignore_index=True
)

print(f"groups: {len(groups)}")
print("size counts:", dict(sorted(Counter(groups["size"]).items())))
print(
    f"path: {(groups['path']).sum()}  "
    f"(≥3 entities: {((groups['path']) & (groups['size'] >= 3)).sum()})  |  "
    f"non-linear: {(~groups['path']).sum()}"
)

# %% [markdown]
# ## By depth

# %%
depth_summary = (
    groups.groupby("depth", dropna=False)
    .agg(n_groups=("size", "size"), size_min=("size", "min"), size_max=("size", "max"))
    .reset_index()
)
depth_summary["sizes"] = depth_summary["depth"].map(
    lambda d: dict(sorted(Counter(groups.loc[groups["depth"] == d, "size"]).items()))
)
print(depth_summary.to_string(index=False))

# %% [markdown]
# ## Notable groups
#
# Non-linear groups and any group with depth ≥ 3 (show longest ascent).

# %%
notable = groups[(~groups["path"]) | (groups["depth"] >= 3)]
for _, s in notable.iterrows():
    shape = "path" if s["path"] else "branch"
    print(f"depth={s['depth']}  size={s['size']}  {shape}  {s['observed']}")
    if s["depth"] and s["depth"] >= 3:
        for ent in s["ascent"]:
            print(f"  {label(ent)}")
    elif not s["path"]:
        hubs = sorted(
            ((d, ent) for ent, d in s["H"].degree() if d >= 3), reverse=True
        )
        for d, ent in hubs[:3]:
            print(f"  hub deg={d}: {label(ent)}")
