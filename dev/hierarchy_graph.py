# %% [markdown]
# Intervention hierarchy graph
#
# Loads a NAV `config.json`, resolves `default_sources` / `generate_defaults`
# the same way scoring does, and builds a digraph of intervention entities.
#
# Entity = `(fixes, observed)`. An edge **A → B** means A is a hierarchical
# subset of B (more general → more specific): when scoring B, agreements at A
# multiply in via an *incoming* ancestor edge. After building the full subset
# closure, transitive reduction keeps only consecutive-depth covering edges.
#
# **Depth** is the longest path from a root in the pruned DAG (not the raw
# intervention count). **Printed fixes** on a non-root omit assignments already
# shared by all immediate predecessors.

# %%
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from boolmore.evaluation.nav_score import _is_intervention_subset
from boolmore.io.load import load_json_config
from boolmore.io.paths import project_root
from boolmore.run_setup import build_evaluator
from boolmore.types import Assignment

# %%
# CONFIG = project_root() / "case_study" / "breast_cancer" / "BreastCancer_config.json"
CONFIG = project_root() / "case_study" / "ABA" / "data" / "ABA_A.json"

# %% [markdown]
# ## Load config and experiments

# %%
config = load_json_config(CONFIG)
assert config["run_type"] == "NAV", f"expected NAV, got {config['run_type']!r}"

evaluator = build_evaluator(config)
exps = evaluator.experiments
default_sources = evaluator.default_sources

print(f"config: {CONFIG.relative_to(project_root())}")
print(f"experiments: {len(exps)}")
print(f"hierarchy (scoring): {evaluator.hierarchy}")
print(f"generate_defaults: {evaluator.generate_defaults}")
print(f"default_sources (resolved): {default_sources}")
if not evaluator.hierarchy:
    print(
        "note: hierarchy scoring is off — flat scores ignore subset edges; "
        "the graph below is still the structural partial order."
    )

# %% [markdown]
# ## Build full subset digraph
#
# Nodes are co-located experiment groups sharing `(fixes, observed)`.
# Edges only within the same observed node.


# %%
def format_fixes(fixes: Assignment) -> str:
    if not fixes:
        return "(none)"
    return ",".join(f"{n}={v}" for n, v in fixes)


def assignment_delta(child: Assignment, shared: dict[str, int]) -> Assignment:
    """Return child settings that are not already present in ``shared``."""
    return tuple((n, v) for n, v in child if shared.get(n) != v)


def predecessor_intersection(H: nx.DiGraph, entity: Entity) -> dict[str, int]:
    preds = list(H.predecessors(entity))
    if not preds:
        return {}
    shared = dict(preds[0][0])
    for pred in preds[1:]:
        other = dict(pred[0])
        shared = {n: v for n, v in shared.items() if other.get(n) == v}
    return shared


Entity = tuple[Assignment, str]


def display_fixes(H: nx.DiGraph, entity: Entity) -> str:
    """Full fixes for roots; otherwise only settings beyond all immediate preds."""
    if H.in_degree(entity) == 0:
        return format_fixes(entity[0])
    return format_fixes(assignment_delta(entity[0], predecessor_intersection(H, entity)))

entity_to_ids: dict[Entity, list[int]] = defaultdict(list)
for exp in exps:
    entity_to_ids[(exp.fixes, exp.observed)].append(exp.id)

for ids in entity_to_ids.values():
    ids.sort()

G = nx.DiGraph()
for entity, ids in entity_to_ids.items():
    fixes, observed = entity
    G.add_node(
        entity,
        ids=ids,
        fixes_str=format_fixes(fixes),
        observed=observed,
    )

by_observed: dict[str, list[Entity]] = defaultdict(list)
for entity in G.nodes:
    by_observed[entity[1]].append(entity)

for observed, entities in by_observed.items():
    for a, b in combinations(entities, 2):
        a_fixes = dict(a[0])
        b_fixes = dict(b[0])
        if _is_intervention_subset(a_fixes, b_fixes, default_sources):
            G.add_edge(a, b)
        if _is_intervention_subset(b_fixes, a_fixes, default_sources):
            G.add_edge(b, a)

print(f"nodes: {G.number_of_nodes()}  |  full subset edges: {G.number_of_edges()}")
print(f"observed nodes: {len(by_observed)}")
print(f"co-located entities (≥2 ids): {sum(1 for ids in entity_to_ids.values() if len(ids) > 1)}")

# %% [markdown]
# ## Transitive reduction + graph depth
#
# Drop shortcut edges when a longer path already exists. Depth is the longest
# path from a root in the pruned DAG (roots = 0).

# %%
G_pruned = nx.transitive_reduction(G)
for node, data in G.nodes(data=True):
    G_pruned.nodes[node].update(data)

for node in nx.topological_sort(G_pruned):
    preds = list(G_pruned.predecessors(node))
    if not preds:
        G_pruned.nodes[node]["depth"] = 0
    else:
        G_pruned.nodes[node]["depth"] = 1 + max(
            G_pruned.nodes[p]["depth"] for p in preds
        )
    G_pruned.nodes[node]["label"] = display_fixes(G_pruned, node)
    ancs = nx.ancestors(G_pruned, node)
    G_pruned.nodes[node]["n_anc"] = len(ancs)
    G_pruned.nodes[node]["n_anc_ids"] = sum(
        len(G_pruned.nodes[a]["ids"]) for a in ancs
    )

depth_counts = Counter(data["depth"] for _, data in G_pruned.nodes(data=True))
print(
    f"after prune: {G_pruned.number_of_nodes()} nodes, "
    f"{G_pruned.number_of_edges()} edges "
    f"(removed {G.number_of_edges() - G_pruned.number_of_edges()} shortcuts)"
)
print(f"graph depth histogram: {dict(sorted(depth_counts.items()))}")

# %% [markdown]
# ## Ancestor-id histogram
#
# For each experiment id, count how many experiment ids sit on its ancestor
# entities (co-located ids on the same ancestor node all count).

# %%
anc_id_counts: list[int] = []
for _, data in G_pruned.nodes(data=True):
    anc_id_counts.extend([data["n_anc_ids"]] * len(data["ids"]))

anc_id_hist = Counter(anc_id_counts)
print(f"experiments: {len(anc_id_counts)}")
print(f"ancestor-id count histogram: {dict(sorted(anc_id_hist.items()))}")

fig, ax = plt.subplots(figsize=(8, 4))
xs = sorted(anc_id_hist)
ys = [anc_id_hist[x] for x in xs]
ax.bar(xs, ys, width=0.8, color="#5b8fa8")
ax.set_xlabel("number of ancestor experiment ids")
ax.set_ylabel("number of experiments")
ax.set_title("Ancestor-id count per experiment")
ax.set_xticks(xs)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Summaries
#
# Non-root `fixes` / predecessor labels omit assignments already shared with
# all immediate predecessors. Full absolute strings stay on `fixes_str` / `table`.

# %%
def _short(text: str, max_len: int = 56) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


rows = []
for entity, data in G_pruned.nodes(data=True):
    pred_labels = [
        G_pruned.nodes[p]["label"]
        for p in sorted(
            G_pruned.predecessors(entity),
            key=lambda p: G_pruned.nodes[p]["label"],
        )
    ]
    rows.append(
        {
            "observed": data["observed"],
            "depth": data["depth"],
            "n_anc": data["n_anc"],
            "n_anc_ids": data["n_anc_ids"],
            "n_pred": len(pred_labels),
            "ids": ",".join(str(i) for i in data["ids"]),
            "fixes": data["label"],
            "fixes_full": data["fixes_str"],
            "predecessors": "; ".join(pred_labels),
            "entity": entity,
        }
    )

table = pd.DataFrame(rows).sort_values(
    ["observed", "depth", "fixes"], ignore_index=True
)

obs_summary = (
    table.groupby("observed", sort=False)
    .agg(
        n=("depth", "size"),
        depth_min=("depth", "min"),
        depth_max=("depth", "max"),
        max_anc_ids=("n_anc_ids", "max"),
    )
    .reset_index()
)
linked = obs_summary[obs_summary["max_anc_ids"] > 0].sort_values(
    ["n", "observed"], ascending=[False, True], ignore_index=True
)
isolated = int((obs_summary["max_anc_ids"] == 0).sum())
print(f"per observed ({len(linked)} with edges, {isolated} isolated omitted):")
print(linked.to_string(index=False))

display = (
    table.drop(columns=["entity", "predecessors", "fixes_full"])
    .assign(fixes=lambda df: df["fixes"].map(_short))
    .sort_values(
        ["n_anc_ids", "depth", "observed"],
        ascending=[False, False, True],
        ignore_index=True,
    )
)

PRINT_TOP = 40
print()
print(
    f"top {min(PRINT_TOP, len(display))} / {len(display)} by ancestor-id count "
    f"(delta fixes truncated; full rows in `table`):"
)
print(display.head(PRINT_TOP).to_string(index=False))

# %% [markdown]
# ## Draw focus node
#
# Pick the entity with the most ancestors. Draw only that node and its
# ancestor cone: nodes labeled by experiment ids, edges by the added fixes.
# The text walkthrough starts at root ancestors (full fixes), then lists
# each covering edge and its delta down to the focus.

# %%
EDGE_LABEL_LEN = 36


def ids_str(H: nx.DiGraph, entity: Entity) -> str:
    return ",".join(str(i) for i in H.nodes[entity]["ids"])


def edge_delta(pred: Entity, succ: Entity) -> str:
    return format_fixes(assignment_delta(succ[0], dict(pred[0])))


def draw_focus_cone(H: nx.DiGraph, focus: Entity, *, max_edge_label: int = EDGE_LABEL_LEN) -> None:
    nodes = {focus} | set(nx.ancestors(H, focus))
    sub = H.subgraph(nodes).copy()
    data = H.nodes[focus]

    roots = sorted(
        (n for n in sub.nodes if sub.in_degree(n) == 0),
        key=lambda n: (H.nodes[n]["depth"], ids_str(H, n)),
    )
    print(
        f"focus: observed={data['observed']}  depth={data['depth']}  "
        f"ids={data['ids']}  ancestors={data['n_anc']}"
    )
    print("roots (full fixes):")
    for root in roots:
        rd = H.nodes[root]
        print(f"  [{ids_str(H, root)}]  {rd['fixes_str']}")

    print("edges (predecessor → successor +delta):")
    edge_list = sorted(
        sub.edges(),
        key=lambda e: (
            H.nodes[e[0]]["depth"],
            ids_str(H, e[0]),
            H.nodes[e[1]]["depth"],
            ids_str(H, e[1]),
        ),
    )
    for pred, succ in edge_list:
        delta = edge_delta(pred, succ)
        print(f"  [{ids_str(H, pred)}] → [{ids_str(H, succ)}]  +{delta}")

    print(f"focus full fixes: {data['fixes_str']}")

    by_depth: dict[int, list[Entity]] = defaultdict(list)
    for node in sub.nodes:
        by_depth[sub.nodes[node]["depth"]].append(node)

    pos: dict[Entity, tuple[float, float]] = {}
    for depth, nodes_at_depth in sorted(by_depth.items()):
        nodes_at_depth = sorted(nodes_at_depth, key=lambda n: ids_str(H, n))
        n = len(nodes_at_depth)
        for i, node in enumerate(nodes_at_depth):
            pos[node] = (i - (n - 1) / 2, -depth)

    node_labels = {node: ids_str(H, node) for node in sub.nodes}
    edge_labels = {
        (pred, succ): _short(edge_delta(pred, succ), max_edge_label)
        for pred, succ in sub.edges()
    }

    colors = ["#f4a261" if n == focus else "#cde4f5" for n in sub.nodes]
    depth_span = max(by_depth) - min(by_depth) + 1
    fig_w = max(8, min(16, 2 + sub.number_of_nodes() * 0.9))
    fig_h = max(4, min(10, 1.5 + depth_span * 1.2))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    nx.draw_networkx_nodes(sub, pos, ax=ax, node_size=1200, node_color=colors)
    nx.draw_networkx_edges(
        sub,
        pos,
        ax=ax,
        arrows=True,
        arrowsize=16,
        edge_color="#555555",
        connectionstyle="arc3,rad=0.05",
    )
    nx.draw_networkx_labels(sub, pos, labels=node_labels, ax=ax, font_size=8)
    nx.draw_networkx_edge_labels(
        sub,
        pos,
        edge_labels=edge_labels,
        ax=ax,
        font_size=6,
        rotate=False,
        label_pos=0.45,
    )
    ax.set_axis_off()
    ax.set_title(
        f"Ancestor cone → {data['observed']}  (ids {data['ids']})",
        fontsize=11,
    )
    fig.tight_layout()
    plt.show()


focus = max(
    G_pruned.nodes,
    key=lambda n: (
        G_pruned.nodes[n]["n_anc_ids"],
        G_pruned.nodes[n]["n_anc"],
        G_pruned.nodes[n]["depth"],
        G_pruned.in_degree(n),
        G_pruned.nodes[n]["label"],
    ),
)
draw_focus_cone(G_pruned, focus)
