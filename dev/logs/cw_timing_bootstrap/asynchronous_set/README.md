# Cubewalkers timing + bootstrap SE (`asynchronous_set`)

Single no-intervention ensemble on the breast-cancer gen108 `.bnet`.
Split wall time (`t_setup` / `t_simulate` / `t_reduce`) and per-node
bootstrap SE of the time-window NAV; sweeps `n_walkers`, `N_TIME_STEPS`,
and `T_WINDOW`.

- **Update scheme:** `asynchronous_set`
- **set_update_prob:** `0.5`
- **Started (UTC):** 2026-10-23T19:18:14.600692+00:00
- **Finished (UTC):** 2026-10-23T19:18:52.437484+00:00
- **Wall clock:** 37.8s
- **SMOKE:** `False`
- **bnet:** `BreastCancer_primes_parsimonious_20261016_173102_9916_gen108.bnet`
- **nodes:** 191
- **Baseline:** W=2048, T=800, window=200, N_BOOT=1000, seed=0
- **Sweep base:** W=2048, T=800, window=200
- **Grids:** walkers=[64, 128, 256, 512, 1024, 2048, 4096], T=[200, 400, 600, 800, 1000], window=[5, 10, 25, 50, 100, 200, 400]

## Baseline

- Timings: `setup=0.146s  simulate=0.293s  reduce=0.010s  total=0.450s`
- median SE: `0.00000`
- max SE: `0.01151`
- mean of node means: `0.3751`

## Sweep takeaways

- n_walkers: simulate scales ~3.6x over grid; median SE shrinks ~0.0x; mean setup share of total=41%; last step vs W2048: med|Δ|=0.00000 max|Δ|=0.03125
- n_time_steps: simulate scales ~0.9x over grid; median SE shrinks ~455637671.7x; mean setup share of total=33%; last step vs T800: med|Δ|=0.00000 max|Δ|=0.02930
- t_window: simulate scales ~1.1x over grid; median SE shrinks ~0.0x; mean setup share of total=35%; last step vs win200: med|Δ|=0.00000 max|Δ|=0.03223

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
