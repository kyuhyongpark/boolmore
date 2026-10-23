# Cubewalkers timing + bootstrap SE (`synchronous`)

Single no-intervention ensemble on the breast-cancer gen108 `.bnet`.
Split wall time (`t_setup` / `t_simulate` / `t_reduce`) and per-node
bootstrap SE of the time-window NAV; sweeps `n_walkers`, `N_TIME_STEPS`,
and `T_WINDOW`.

- **Update scheme:** `synchronous`
- **Started (UTC):** 2026-10-23T19:15:58.487066+00:00
- **Finished (UTC):** 2026-10-23T19:16:51.907445+00:00
- **Wall clock:** 53.4s
- **SMOKE:** `False`
- **bnet:** `BreastCancer_primes_parsimonious_20261016_173102_9916_gen108.bnet`
- **nodes:** 191
- **Baseline:** W=2048, T=800, window=200, N_BOOT=1000, seed=0
- **Sweep base:** W=2048, T=800, window=200
- **Grids:** walkers=[64, 128, 256, 512, 1024, 2048, 4096], T=[200, 400, 600, 800, 1000], window=[5, 10, 25, 50, 100, 200, 400]

## Baseline

- Timings: `setup=0.143s  simulate=0.229s  reduce=0.008s  total=0.380s`
- median SE: `0.00000`
- max SE: `0.01157`
- mean of node means: `0.3747`

## Sweep takeaways

- n_walkers: simulate scales ~6.3x over grid; median SE shrinks ~0.0x; mean setup share of total=49%; last step vs W2048: med|Δ|=0.00000 max|Δ|=0.02832
- n_time_steps: simulate scales ~1.1x over grid; median SE shrinks ~129626345.8x; mean setup share of total=8%; last step vs T800: med|Δ|=0.00000 max|Δ|=0.04297
- t_window: simulate scales ~1.0x over grid; median SE shrinks ~0.0x; mean setup share of total=8%; last step vs win200: med|Δ|=0.00000 max|Δ|=0.04053

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
