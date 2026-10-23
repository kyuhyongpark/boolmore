# ABA-A scores: cubewalkers (GT) vs `min_traps` + LDOI

Comparison of NAV total scores on every `.bnet` under `case_study/ABA`,
using experiments from `case_study/ABA/data/ABA_A.json` (`data_A.csv`, `default_sources`).

Ground truth: `cubewalkers`. Compared approximate methods:

- `min_traps`
- `ldoi`
- `enhanced(c=1)`
- `enhanced(c=2)`
- `enhanced+j(c=2)`

- **Data set:** `A`
- **Started (UTC):** 2026-10-23T21:03:34.562772+00:00
- **Finished (UTC):** 2026-10-23T21:04:52.486080+00:00
- **Wall clock:** 77.9s
- **Hierarchy:** True
- **Experiments:** 505
- **Unique fixes:** 260
- **Cubewalkers (GT):** `asynchronous_set`, W=2048, T=600, T_window=120, set_update_prob=0.5
- **Bnets scanned:** 29
- **All methods OK:** 22
- **Failed:** 7
- **min_traps cache:** `use=True`, `force_recompute=False`, hits=29, misses=0
- **cw cache:** `use=True`, `force_recompute=False`, hits=22, misses=0

## Score agreement vs cubewalkers

| method | Pearson r | Spearman ρ | pairwise order | mean |Δrank| | mean Δscore |
| --- | ---: | ---: | ---: | ---: | ---: |
| `min_traps` | `0.4765` | `0.4613` | `0.6710` (155/231) | `5.273` | `61.241` |
| `ldoi` | `0.3287` | `0.3055` | `0.5801` (134/231) | `6.091` | `-192.162` |
| `enhanced(c=1)` | `0.2929` | `0.1733` | `0.5498` (127/231) | `6.727` | `-190.644` |
| `enhanced(c=2)` | `0.2414` | `0.4274` | `0.6623` (153/231) | `5.273` | `-52.406` |
| `enhanced+j(c=2)` | `0.2402` | `0.4274` | `0.6623` (153/231) | `5.273` | `-52.386` |
- `min_traps` overlaps: top-1=0.000, top-3=0.333, top-5=0.200
- `ldoi` overlaps: top-1=0.000, top-3=0.000, top-5=0.400
- `enhanced(c=1)` overlaps: top-1=0.000, top-3=0.000, top-5=0.400
- `enhanced(c=2)` overlaps: top-1=0.000, top-3=0.333, top-5=0.400
- `enhanced+j(c=2)` overlaps: top-1=0.000, top-3=0.333, top-5=0.400

## Method timings

Wall time for prediction + NAV scoring + model score (models where that method completed). `min_traps` and `cubewalkers` times reuse the originally measured values when loaded from cache. LDOI always runs fresh. Cubewalkers is skipped when earlier methods fail.

| method | n | total (s) | mean (s) | median (s) | min (s) | max (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `cubewalkers` | 22 | 1779.76 | 80.90 | 80.21 | 68.99 | 97.64 |
| `min_traps` | 22 | 49.19 | 2.24 | 2.15 | 1.97 | 2.69 |
| `ldoi` | 22 | 2.28 | 0.10 | 0.10 | 0.09 | 0.12 |
| `enhanced(c=1)` | 22 | 6.56 | 0.30 | 0.28 | 0.21 | 0.43 |
| `enhanced(c=2)` | 22 | 16.08 | 0.73 | 0.73 | 0.52 | 0.96 |
| `enhanced+j(c=2)` | 22 | 17.67 | 0.80 | 0.81 | 0.52 | 1.11 |

## Failures

- `baseline_models/ABA_2006_Li.bnet`: min_traps: ValueError: 8-nitro-cGMP is not in the model
- `baseline_models/ABA_2018_Waidyarathne.bnet`: min_traps: ValueError: 8-nitro-cGMP is not in the model
- `baseline_models/ABA_2019_Maheshwari.bnet`: min_traps: ValueError: 8-nitro-cGMP is not in the model
- `baseline_models/ABA_GA_base_B.bnet`: min_traps: ValueError: Ca2c is not in the model
- `generated_models/GA0_B.bnet`: min_traps: ValueError: Ca2c is not in the model
- `generated_models/GA1_B.bnet`: min_traps: ValueError: Ca2c is not in the model
- `generated_models/GA2_B.bnet`: min_traps: ValueError: Ca2c is not in the model

## Artifacts

Stable filenames (overwritten on each run):

- [`compare_aba_scores.log`](compare_aba_scores.log) — full run log
- [`min_traps_cache.json`](min_traps_cache.json) — `min_traps` score + timing cache
- [`cw_cache.json`](cw_cache.json) — `cubewalkers` score + timing cache
- [`method_timings.csv`](method_timings.csv) — timing summary CSV
- [`method_timings_per_model.csv`](method_timings_per_model.csv) — per-model wall times
- [`compare_aba_scores_scatter.png`](compare_aba_scores_scatter.png)
- [`compare_aba_ranks_scatter.png`](compare_aba_ranks_scatter.png)
- [`compare_aba_score_diff.png`](compare_aba_score_diff.png)
- [`compare_aba_method_timings.png`](compare_aba_method_timings.png)
