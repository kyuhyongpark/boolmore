# ABA-A scores: `min_traps` vs LDOI family

Comparison of NAV total scores on every `.bnet` under `case_study/ABA`,
using experiments from `case_study/ABA/data/ABA_A.json` (`data_A.csv`, `default_sources`).

Reference method: `min_traps`. Compared LDOI variants:

- `ldoi`
- `enhanced(c=1)`
- `enhanced(c=2)`
- `enhanced+j(c=2)`

- **Data set:** `A`
- **Started (UTC):** 2026-10-23T18:58:39.862737+00:00
- **Finished (UTC):** 2026-10-23T19:00:11.319816+00:00
- **Wall clock:** 91.5s
- **Hierarchy:** True
- **Experiments:** 505
- **Bnets scanned:** 29
- **All methods OK:** 22
- **Failed:** 7
- **min_traps cache:** `use=True`, `force_recompute=False`, hits=29, misses=0

## Score agreement vs min_traps

| method | Pearson r | Spearman ρ | pairwise order | mean |Δrank| | mean Δscore |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ldoi` | `0.1660` | `0.1101` | `0.5368` (124/231) | `7.273` | `-253.403` |
| `enhanced(c=1)` | `0.2171` | `0.1112` | `0.5498` (127/231) | `7.364` | `-251.885` |
| `enhanced(c=2)` | `0.6221` | `0.3540` | `0.6277` (145/231) | `5.545` | `-113.647` |
| `enhanced+j(c=2)` | `0.6219` | `0.3540` | `0.6277` (145/231) | `5.545` | `-113.627` |
- `ldoi` overlaps: top-1=0.000, top-3=0.000, top-5=0.200
- `enhanced(c=1)` overlaps: top-1=0.000, top-3=0.000, top-5=0.200
- `enhanced(c=2)` overlaps: top-1=0.000, top-3=0.000, top-5=0.000
- `enhanced+j(c=2)` overlaps: top-1=0.000, top-3=0.000, top-5=0.000

## Method timings

Wall time for prediction + NAV scoring + model score (models where that method completed). `min_traps` times reuse the originally measured values when loaded from cache.

| method | n | total (s) | mean (s) | median (s) | min (s) | max (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `min_traps` | 22 | 61.14 | 2.78 | 2.74 | 2.32 | 3.40 |
| `ldoi` | 22 | 2.89 | 0.13 | 0.12 | 0.11 | 0.25 |
| `enhanced(c=1)` | 22 | 8.16 | 0.37 | 0.36 | 0.24 | 0.56 |
| `enhanced(c=2)` | 22 | 19.38 | 0.88 | 0.85 | 0.64 | 1.30 |
| `enhanced+j(c=2)` | 22 | 20.89 | 0.95 | 0.97 | 0.57 | 1.30 |

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
- [`method_timings.csv`](method_timings.csv) — timing summary CSV
- [`method_timings_per_model.csv`](method_timings_per_model.csv) — per-model wall times
- [`compare_aba_scores_scatter.png`](compare_aba_scores_scatter.png)
- [`compare_aba_ranks_scatter.png`](compare_aba_ranks_scatter.png)
- [`compare_aba_score_diff.png`](compare_aba_score_diff.png)
- [`compare_aba_method_timings.png`](compare_aba_method_timings.png)
