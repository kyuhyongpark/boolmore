# ABA-A scores: `min_traps` vs `ldoi`

Comparison of NAV total scores on every `.bnet` under `case_study/ABA`,
using experiments from `case_study/ABA/data/ABA_A.json` (`data_A.csv`, `default_sources`).

- **Data set:** `A`
- **Started (UTC):** 2026-10-23T18:48:11.248016+00:00
- **Finished (UTC):** 2026-10-23T18:48:53.111473+00:00
- **Wall clock:** 41.9s
- **Hierarchy:** True
- **Experiments:** 505
- **Bnets scanned:** 29
- **Both methods OK:** 22
- **Failed:** 7
- **min_traps cache:** `use=True`, `force_recompute=False`, hits=29, misses=0

## Score agreement

- Pearson *r*: `0.1660`
- Spearman *ρ*: `0.1101`
- Pairwise order agreement: `0.5368` (124/231 pairs)
- Mean |rank_min − rank_ldoi|: `7.273`
- Score diff (ldoi − min): mean=`-253.403`, std=`40.154`, min=`-297.882`, max=`-152.951`
- Top-1 set overlap: `0.000`
- Top-3 set overlap: `0.000`
- Top-5 set overlap: `0.200`

## Method timings

Wall time for prediction + NAV scoring + model score (models where that method completed). `min_traps` times reuse the originally measured values when loaded from cache.

| method | n | total (s) | mean (s) | median (s) | min (s) | max (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `min_traps` | 22 | 61.14 | 2.78 | 2.74 | 2.32 | 3.40 |
| `ldoi` | 22 | 3.14 | 0.14 | 0.14 | 0.11 | 0.25 |

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

- [`compare_aba_scores_min_traps_vs_ldoi.log`](compare_aba_scores_min_traps_vs_ldoi.log) — full run log
- [`min_traps_cache.json`](min_traps_cache.json) — `min_traps` score + timing cache
- [`method_timings.csv`](method_timings.csv) — timing summary CSV
- [`method_timings_per_model.csv`](method_timings_per_model.csv) — per-model wall times
- [`compare_aba_scores_scatter.png`](compare_aba_scores_scatter.png)
- [`compare_aba_ranks_scatter.png`](compare_aba_ranks_scatter.png)
- [`compare_aba_score_diff.png`](compare_aba_score_diff.png)
- [`compare_aba_method_timings.png`](compare_aba_method_timings.png)
