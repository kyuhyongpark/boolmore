# Cubewalkers: compare update schemes (NAV + timing)

Single no-intervention ensemble on the breast-cancer gen108 `.bnet`.
Compares `synchronous`, `asynchronous_set`, and `asynchronous` NAV
(bootstrap SE) and split wall times at scheme-specific horizons.

- **Started (UTC):** 2026-10-23T19:40:57.573885+00:00
- **Finished (UTC):** 2026-10-23T19:41:17.166898+00:00
- **Wall clock:** 19.6s
- **SMOKE:** `False`
- **bnet:** `BreastCancer_primes_parsimonious_20261016_173102_9916_gen108.bnet`
- **nodes:** 191
- **W:** 2048
- **N_BOOT:** 1000
- **seed:** 0
- **set_update_prob:** 0.5

## Horizons

- **synchronous:** T=800, T_window=160
- **asynchronous_set:** T=1200, T_window=240
- **asynchronous:** T=16000, T_window=3200

## Per-scheme summary

### `synchronous`

- Timings: `setup=0.137s  simulate=1.549s  reduce=0.008s  total=1.695s`
- mean of node means: `0.3725`
- median SE: `0.00000`
- max SE: `0.01143`

### `asynchronous_set`

- Timings: `setup=0.136s  simulate=1.618s  reduce=0.008s  total=1.762s`
- mean of node means: `0.3745`
- median SE: `0.00000`
- max SE: `0.01148`

### `asynchronous`

- Timings: `setup=0.136s  simulate=5.045s  reduce=0.053s  total=5.234s`
- mean of node means: `0.3727`
- median SE: `0.00000`
- max SE: `0.01184`

## Timing vs synchronous

- **synch:** t_total=`1.695s`  ratio=`1.00x`  Δ=`+0.000s`
- **asynch_set:** t_total=`1.762s`  ratio=`1.04x`  Δ=`+0.068s`
- **asynch:** t_total=`5.234s`  ratio=`3.09x`  Δ=`+3.539s`

## Pairwise ΔNAV

- **asynch_set − synch:** med|Δ|=`0.00000`  mean|Δ|=`0.00923`  max|Δ|=`0.05518`  RMSE=`0.01739`  corr=`0.9989`  medSE(Δ)=`0.00000`
- **asynch − synch:** med|Δ|=`0.00000`  mean|Δ|=`0.00841`  max|Δ|=`0.09961`  RMSE=`0.01583`  corr=`0.9991`  medSE(Δ)=`0.00000`
- **asynch − asynch_set:** med|Δ|=`0.00000`  mean|Δ|=`0.00667`  max|Δ|=`0.05127`  RMSE=`0.01215`  corr=`0.9995`  medSE(Δ)=`0.00000`

## Artifacts

Stable filenames (overwritten on each run):

- [`cw_compare_update_schemes.log`](cw_compare_update_schemes.log) — full run log
- [`README.md`](README.md) — this summary
- [`scheme_nav.csv`](scheme_nav.csv) — per-node NAV / Δ
- [`scheme_timings.csv`](scheme_timings.csv) — timings
- [`cw_compare_schemes_nav_overlay.png`](cw_compare_schemes_nav_overlay.png)
- [`cw_compare_schemes_nav_diff.png`](cw_compare_schemes_nav_diff.png)
- [`cw_compare_schemes_summary.png`](cw_compare_schemes_summary.png)
- [`cw_compare_schemes_nav_scatter.png`](cw_compare_schemes_nav_scatter.png)
