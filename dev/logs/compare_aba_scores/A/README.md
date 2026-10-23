# ABA-A scores: `min_traps` vs LDOI family vs cubewalkers

Comparison of NAV total scores on every `.bnet` under `case_study/ABA`,
using experiments from `case_study/ABA/data/ABA_A.json` (`data_A.csv`, `default_sources`).

Reference method: `min_traps`. Compared methods:

- `ldoi`
- `enhanced(c=1)`
- `enhanced(c=2)`
- `enhanced+j(c=2)`
- `cubewalkers`

- **Data set:** `A`
- **Started (UTC):** 2026-10-23T20:04:11.187349+00:00
- **Finished (UTC):** 2026-10-23T20:35:59.697580+00:00
- **Wall clock:** 1908.5s
- **Hierarchy:** True
- **Experiments:** 505
- **Unique fixes:** 260
- **Cubewalkers:** `asynchronous_set`, W=2048, T=600, T_window=120, set_update_prob=0.5
- **Bnets scanned:** 29
- **All methods OK:** 22
- **Failed:** 7
- **min_traps cache:** `use=True`, `force_recompute=False`, hits=0, misses=29
- **cw cache:** `use=True`, `force_recompute=False`, hits=0, misses=22

## Score agreement vs min_traps

| method | Pearson r | Spearman ρ | pairwise order | mean |Δrank| | mean Δscore |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ldoi` | `0.1660` | `0.1101` | `0.5368` (124/231) | `7.273` | `-253.403` |
| `enhanced(c=1)` | `0.2171` | `0.1112` | `0.5498` (127/231) | `7.364` | `-251.885` |
| `enhanced(c=2)` | `0.6221` | `0.3540` | `0.6277` (145/231) | `5.545` | `-113.647` |
| `enhanced+j(c=2)` | `0.6219` | `0.3540` | `0.6277` (145/231) | `5.545` | `-113.627` |
| `cubewalkers` | `0.4765` | `0.4613` | `0.6710` (155/231) | `5.273` | `-61.241` |
- `ldoi` overlaps: top-1=0.000, top-3=0.000, top-5=0.200
- `enhanced(c=1)` overlaps: top-1=0.000, top-3=0.000, top-5=0.200
- `enhanced(c=2)` overlaps: top-1=0.000, top-3=0.000, top-5=0.000
- `enhanced+j(c=2)` overlaps: top-1=0.000, top-3=0.000, top-5=0.000
- `cubewalkers` overlaps: top-1=0.000, top-3=0.333, top-5=0.200

## Method timings

Wall time for prediction + NAV scoring + model score (models where that method completed). `min_traps` and `cubewalkers` times reuse the originally measured values when loaded from cache. Cubewalkers is skipped when earlier methods fail.

| method | n | total (s) | mean (s) | median (s) | min (s) | max (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `min_traps` | 22 | 49.19 | 2.24 | 2.15 | 1.97 | 2.69 |
| `ldoi` | 22 | 2.20 | 0.10 | 0.10 | 0.09 | 0.11 |
| `enhanced(c=1)` | 22 | 6.58 | 0.30 | 0.28 | 0.21 | 0.52 |
| `enhanced(c=2)` | 22 | 16.04 | 0.73 | 0.70 | 0.53 | 1.01 |
| `enhanced+j(c=2)` | 22 | 17.79 | 0.81 | 0.80 | 0.51 | 1.15 |
| `cubewalkers` | 22 | 1779.76 | 80.90 | 80.21 | 68.99 | 97.64 |

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
