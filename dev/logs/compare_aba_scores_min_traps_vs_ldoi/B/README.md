# ABA-B scores: `min_traps` vs `ldoi`

Comparison of NAV total scores on every `.bnet` under `case_study/ABA`,
using experiments from `case_study/ABA/data/ABA_B.json` (`data_B.csv`, `default_sources`).

- **Data set:** `B`
- **Started (UTC):** 2026-10-23T18:51:12.159770+00:00
- **Finished (UTC):** 2026-10-23T18:51:49.689100+00:00
- **Wall clock:** 37.5s
- **Hierarchy:** True
- **Experiments:** 505
- **Bnets scanned:** 29
- **Both methods OK:** 4
- **Failed:** 25
- **min_traps cache:** `use=True`, `force_recompute=False`, hits=29, misses=0

## Score agreement

- Pearson *r*: `0.9654`
- Spearman *ρ*: `1.0000`
- Pairwise order agreement: `1.0000` (6/6 pairs)
- Mean |rank_min − rank_ldoi|: `0.000`
- Score diff (ldoi − min): mean=`-25.119`, std=`32.641`, min=`-68.719`, max=`0.000`
- Top-1 set overlap: `1.000`
- Top-3 set overlap: `1.000`

## Method timings

Wall time for prediction + NAV scoring + model score (models where that method completed). `min_traps` times reuse the originally measured values when loaded from cache.

| method | n | total (s) | mean (s) | median (s) | min (s) | max (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `min_traps` | 4 | 10.48 | 2.62 | 2.61 | 2.51 | 2.74 |
| `ldoi` | 4 | 0.64 | 0.16 | 0.16 | 0.12 | 0.21 |

## Failures

- `baseline_models/ABA_2006_Li.bnet`: min_traps: ValueError: 8-nitro-cGMP is not in the model
- `baseline_models/ABA_2017_Albert.bnet`: min_traps: ValueError: Ca2osc is not in the model
- `baseline_models/ABA_2017_VA-KEV_del.bnet`: min_traps: ValueError: Ca2osc is not in the model
- `baseline_models/ABA_2018_Waidyarathne.bnet`: min_traps: ValueError: 8-nitro-cGMP is not in the model
- `baseline_models/ABA_2019_Maheshwari.bnet`: min_traps: ValueError: 8-nitro-cGMP is not in the model
- `baseline_models/ABA_GA_base_A.bnet`: min_traps: ValueError: Ca2osc is not in the model
- `generated_models/GA0_A.bnet`: min_traps: ValueError: Ca2osc is not in the model
- `generated_models/GA1_A.bnet`: min_traps: ValueError: Ca2osc is not in the model
- `generated_models/GA2_A.bnet`: min_traps: ValueError: Ca2osc is not in the model
- `generated_models/GA_VA-KEV_del.bnet`: min_traps: ValueError: Ca2osc is not in the model
- `generated_models/repetitions/20241007143135_9306_gen124.bnet`: min_traps: ValueError: Ca2osc is not in the model
- `generated_models/repetitions/20241009072111_9984_gen68.bnet`: min_traps: ValueError: Ca2osc is not in the model
- `generated_models/repetitions/20241009211257_9880_gen154.bnet`: min_traps: ValueError: Ca2osc is not in the model
- `generated_models/repetitions/20241010082016_9453_gen140.bnet`: min_traps: ValueError: Ca2osc is not in the model
- `generated_models/repetitions/20241011222037_9848_gen50.bnet`: min_traps: ValueError: Ca2osc is not in the model
- `generated_models/repetitions/20241014130450_9902_gen56.bnet`: min_traps: ValueError: Ca2osc is not in the model
- `generated_models/repetitions/rep_20230917_7545_gen112.bnet`: min_traps: ValueError: Ca2osc is not in the model
- `generated_models/repetitions/rep_20230918_7560_gen105.bnet`: min_traps: ValueError: Ca2osc is not in the model
- `generated_models/repetitions/rep_20230919_8030_gen113.bnet`: min_traps: ValueError: Ca2osc is not in the model
- `generated_models/repetitions/rep_20230920_7788_gen121.bnet`: min_traps: ValueError: Ca2osc is not in the model
- `generated_models/repetitions/rep_20230921_7687_gen104.bnet`: min_traps: ValueError: Ca2osc is not in the model
- `generated_models/repetitions/rep_20230922_8036_gen125.bnet`: min_traps: ValueError: Ca2osc is not in the model
- `generated_models/repetitions/rep_20230923_8025_gen141.bnet`: min_traps: ValueError: Ca2osc is not in the model
- `generated_models/repetitions/rep_20230924_8037_gen129.bnet`: min_traps: ValueError: Ca2osc is not in the model
- `generated_models/repetitions/rep_20230925_7821_gen113.bnet`: min_traps: ValueError: Ca2osc is not in the model

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
