# Cubewalkers timing + bootstrap SE (`asynchronous`)

Single no-intervention ensemble on the breast-cancer gen108 `.bnet`.
Split wall time (`t_setup` / `t_simulate` / `t_reduce`) and per-node
bootstrap SE of the time-window NAV; sweeps `n_walkers`, `N_TIME_STEPS`,
and `T_WINDOW`.

- **Update scheme:** `asynchronous`
- **Started (UTC):** 2026-10-23T19:11:13.043277+00:00
- **Finished (UTC):** 2026-10-23T19:12:23.150423+00:00
- **Wall clock:** 70.1s
- **SMOKE:** `False`
- **bnet:** `BreastCancer_primes_parsimonious_20261016_173102_9916_gen108.bnet`
- **nodes:** 191
- **Baseline:** W=2048, T=8000, window=2000, N_BOOT=1000, seed=0
- **Sweep base:** W=2048, T=8000, window=2000
- **Grids:** walkers=[64, 128, 256, 512, 1024, 2048, 4096], T=[2000, 4000, 6000, 8000, 10000], window=[50, 100, 250, 500, 1000, 2000, 4000]

## Baseline

- Timings: `setup=0.149s  simulate=2.177s  reduce=0.037s  total=2.364s`
- median SE: `0.00000`
- max SE: `0.01141`
- mean of node means: `0.3761`

## Sweep takeaways

- n_walkers: simulate scales ~1.3x over grid; median SE shrinks ~0.0x; mean setup share of total=6%; last step vs W2048: med|Δ|=0.00000 max|Δ|=0.02808
- n_time_steps: simulate scales ~3.3x over grid; median SE shrinks ~5046430762.6x; mean setup share of total=8%; last step vs T8000: med|Δ|=0.00000 max|Δ|=0.03369
- t_window: simulate scales ~1.2x over grid; median SE shrinks ~0.0x; mean setup share of total=6%; last step vs win2000: med|Δ|=0.00000 max|Δ|=0.03564

Note: first launch after Model build may include NVRTC compile in
`t_simulate`; warm-up reduces but does not eliminate per-Model name
rebuild cost.

## Artifacts

Stable filenames (overwritten on each run):

- [`cw_timing_bootstrap.log`](cw_timing_bootstrap.log) — full run log
- [`README.md`](README.md) — this summary
- [`cw_timing_bootstrap_baseline_nav.png`](cw_timing_bootstrap_baseline_nav.png)
- [`cw_timing_bootstrap_sweep_walkers.png`](cw_timing_bootstrap_sweep_walkers.png)
- [`cw_timing_bootstrap_sweep_T.png`](cw_timing_bootstrap_sweep_T.png)
- [`cw_timing_bootstrap_sweep_window.png`](cw_timing_bootstrap_sweep_window.png)
- [`cw_timing_bootstrap_sweep_walkers_nav.png`](cw_timing_bootstrap_sweep_walkers_nav.png)
- [`cw_timing_bootstrap_sweep_T_nav.png`](cw_timing_bootstrap_sweep_T_nav.png)
- [`cw_timing_bootstrap_sweep_window_nav.png`](cw_timing_bootstrap_sweep_window_nav.png)
