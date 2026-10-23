# ABA-B scores: `min_traps` vs LDOI family vs cubewalkers

Comparison of NAV total scores on every `.bnet` under `case_study/ABA`,
using experiments from `case_study/ABA/data/ABA_B.json` (`data_B.csv`, `default_sources`).

Reference method: `min_traps`. Compared methods:

- `ldoi`
- `enhanced(c=1)`
- `enhanced(c=2)`
- `enhanced+j(c=2)`
- `cubewalkers`

- **Data set:** `B`
- **Started (UTC):** 2026-10-23T20:41:22.953025+00:00
- **Finished (UTC):** 2026-10-23T20:47:04.348973+00:00
- **Wall clock:** 341.4s
- **Hierarchy:** True
- **Experiments:** 505
- **Unique fixes:** 261
- **Cubewalkers:** `asynchronous_set`, W=2048, T=600, T_window=120, set_update_prob=0.5
- **Bnets scanned:** 29
- **All methods OK:** 4
- **Failed:** 25
- **min_traps cache:** `use=True`, `force_recompute=False`, hits=0, misses=29
- **cw cache:** `use=True`, `force_recompute=False`, hits=0, misses=4

## Score agreement vs min_traps

| method | Pearson r | Spearman ρ | pairwise order | mean |Δrank| | mean Δscore |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ldoi` | `0.9654` | `1.0000` | `1.0000` (6/6) | `0.000` | `-25.119` |
| `enhanced(c=1)` | `0.9654` | `1.0000` | `1.0000` (6/6) | `0.000` | `-25.119` |
| `enhanced(c=2)` | `0.9663` | `1.0000` | `1.0000` (6/6) | `0.000` | `-25.811` |
| `enhanced+j(c=2)` | `0.9663` | `1.0000` | `1.0000` (6/6) | `0.000` | `-25.811` |
| `cubewalkers` | `0.9960` | `1.0000` | `1.0000` (6/6) | `0.000` | `-6.360` |
- `ldoi` overlaps: top-1=1.000, top-3=1.000
- `enhanced(c=1)` overlaps: top-1=1.000, top-3=1.000
- `enhanced(c=2)` overlaps: top-1=1.000, top-3=1.000
- `enhanced+j(c=2)` overlaps: top-1=1.000, top-3=1.000
- `cubewalkers` overlaps: top-1=1.000, top-3=1.000

## Method timings

Wall time for prediction + NAV scoring + model score (models where that method completed). `min_traps` and `cubewalkers` times reuse the originally measured values when loaded from cache. Cubewalkers is skipped when earlier methods fail.

| method | n | total (s) | mean (s) | median (s) | min (s) | max (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `min_traps` | 4 | 8.46 | 2.11 | 2.12 | 2.04 | 2.17 |
| `ldoi` | 4 | 0.46 | 0.11 | 0.11 | 0.11 | 0.12 |
| `enhanced(c=1)` | 4 | 0.45 | 0.11 | 0.11 | 0.10 | 0.12 |
| `enhanced(c=2)` | 4 | 0.59 | 0.15 | 0.15 | 0.11 | 0.18 |
| `enhanced+j(c=2)` | 4 | 0.58 | 0.14 | 0.15 | 0.11 | 0.17 |
| `cubewalkers` | 4 | 294.19 | 73.55 | 73.12 | 70.41 | 77.55 |

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

- [`compare_aba_scores.log`](compare_aba_scores.log) — full run log
- [`min_traps_cache.json`](min_traps_cache.json) — `min_traps` score + timing cache
- [`cw_cache.json`](cw_cache.json) — `cubewalkers` score + timing cache
- [`method_timings.csv`](method_timings.csv) — timing summary CSV
- [`method_timings_per_model.csv`](method_timings_per_model.csv) — per-model wall times
- [`compare_aba_scores_scatter.png`](compare_aba_scores_scatter.png)
- [`compare_aba_ranks_scatter.png`](compare_aba_ranks_scatter.png)
- [`compare_aba_score_diff.png`](compare_aba_score_diff.png)
- [`compare_aba_method_timings.png`](compare_aba_method_timings.png)
