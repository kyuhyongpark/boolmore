# %% [markdown]
# Bench: PyBoolNet BNetToPrime vs Primer-B ``.bnet`` import
#
# Compares wall time of:
#
# 1. **pyboolnet** — ``bnet_file2primes`` (BNetToPrime subprocess)
# 2. **primer** — ``boolmore.io.bnet_primes.bnet_file_to_primes``
#
# On CAD, breast-cancer parsimonious, and RA ``243.bnet``.
#
# The PyBoolNet arm runs in a child process with a per-network wall-clock
# cutoff so a slow ``243.bnet`` import cannot dominate the session. On
# timeout, remaining PyBoolNet repeats and the equality check for that
# network are skipped; Primer-B still runs to completion.
#
# Cutoffs (seconds), overridable via env:
# ``BOOLMORE_BENCH_BNET_TIMEOUT_CAD``, ``_BREAST``, ``_RA243``.
#
# Prints to stdout and writes a timestamped log under ``dev/logs/``.

# %%
from __future__ import annotations

import os
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter

from boolmore.io.bnet_primes import bnet_file_to_primes, canon_primes

REPO = Path(__file__).resolve().parents[1]
LOG_DIR = Path(__file__).resolve().parent / "logs"

N_WARMUP = 1
N_REPEATS = 3


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


@dataclass(frozen=True)
class NetworkSpec:
    label: str
    path: Path
    pyboolnet_timeout_s: float


NETWORKS = [
    NetworkSpec(
        "CAD",
        REPO / "demo" / "CAD.bnet",
        _env_float("BOOLMORE_BENCH_BNET_TIMEOUT_CAD", 60.0),
    ),
    NetworkSpec(
        "breast_cancer",
        REPO / "case_study" / "breast_cancer" / "parsimonious.bnet",
        _env_float("BOOLMORE_BENCH_BNET_TIMEOUT_BREAST", 60.0),
    ),
    NetworkSpec(
        "RA_243",
        REPO / "case_study" / "RA" / "243.bnet",
        _env_float("BOOLMORE_BENCH_BNET_TIMEOUT_RA243", 30.0),
    ),
]


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def summarize(times: list[float]) -> str:
    mean = statistics.mean(times)
    stdev = statistics.stdev(times) if len(times) > 1 else 0.0
    return (
        f"mean={mean:.6f}s  stdev={stdev:.6f}s  "
        f"min={min(times):.6f}s  max={max(times):.6f}s  "
        f"(n={len(times)})"
    )


def _pyboolnet_worker(path_str: str, conn) -> None:
    """Child entry: import via BNetToPrime and send canonized primes."""
    try:
        from pyboolnet.external.bnet2primes import bnet_file2primes

        primes = bnet_file2primes(path_str)
        conn.send(("ok", canon_primes(primes)))
    except Exception as exc:  # noqa: BLE001 — surface any failure to parent
        conn.send(("error", repr(exc)))
    finally:
        conn.close()


def run_pyboolnet_once(
    path: Path, timeout_s: float
) -> tuple[float | None, dict | None, str | None]:
    """Run one PyBoolNet import with a hard wall-clock cutoff.

    Returns ``(elapsed_s, primes_or_None, status)`` where status is
    ``None`` on success, ``"TIMEOUT"``, or an error string.
    """
    ctx = get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=_pyboolnet_worker,
        args=(str(path), child_conn),
    )
    t0 = perf_counter()
    proc.start()
    child_conn.close()
    proc.join(timeout_s)
    elapsed = perf_counter() - t0

    if proc.is_alive():
        proc.terminate()
        proc.join(5.0)
        if proc.is_alive():
            proc.kill()
            proc.join(5.0)
        parent_conn.close()
        return None, None, "TIMEOUT"

    if parent_conn.poll():
        status, payload = parent_conn.recv()
        parent_conn.close()
        if status == "ok":
            return elapsed, payload, None
        return None, None, str(payload)

    parent_conn.close()
    exitcode = proc.exitcode
    return None, None, f"no_result_exitcode={exitcode}"


def timed_primer(path: Path, *, warmup: int, repeats: int) -> list[float]:
    for _ in range(warmup):
        bnet_file_to_primes(path)
    times: list[float] = []
    for _ in range(repeats):
        t0 = perf_counter()
        bnet_file_to_primes(path)
        times.append(perf_counter() - t0)
    return times


def timed_pyboolnet(
    path: Path,
    timeout_s: float,
    *,
    warmup: int,
    repeats: int,
) -> tuple[list[float], dict | None, bool]:
    """Time PyBoolNet imports; stop early on timeout/error.

    Returns ``(times, first_successful_primes, timed_out)``.
    """
    times: list[float] = []
    first_primes: dict | None = None
    timed_out = False

    # Warmup counts toward cutoff policy: a timeout during warmup aborts.
    for i in range(warmup + repeats):
        elapsed, primes, status = run_pyboolnet_once(path, timeout_s)
        if status == "TIMEOUT":
            print(
                f"  pyboolnet TIMEOUT after {timeout_s:.1f}s "
                f"(attempt {i + 1}/{warmup + repeats})"
            )
            timed_out = True
            break
        if status is not None:
            print(f"  pyboolnet ERROR: {status}")
            timed_out = True
            break
        assert elapsed is not None and primes is not None
        if first_primes is None:
            first_primes = primes
        if i >= warmup:
            times.append(elapsed)
            print(f"  pyboolnet repeat {i - warmup + 1}: {elapsed:.6f}s")

    return times, first_primes, timed_out


# %%
def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / (
        f"bench_bnet_import_primer_"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.log"
    )
    log_file = log_path.open("w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_file)

    print(f"log file: {log_path}")
    print(f"started (UTC): {datetime.now(timezone.utc).isoformat()}")
    print(f"warmup={N_WARMUP}  repeats={N_REPEATS}")
    print()

    for spec in NETWORKS:
        print("=" * 72)
        print(f"network: {spec.label}")
        print(f"  path: {spec.path}")
        print(f"  pyboolnet timeout: {spec.pyboolnet_timeout_s:.1f}s")
        if not spec.path.is_file():
            print("  SKIP: file missing")
            print()
            continue

        print("-- pyboolnet (BNetToPrime) --")
        pb_times, pb_primes, pb_timeout = timed_pyboolnet(
            spec.path,
            spec.pyboolnet_timeout_s,
            warmup=N_WARMUP,
            repeats=N_REPEATS,
        )
        if pb_times:
            print(f"  {summarize(pb_times)}")
        elif pb_timeout:
            print("  no successful timed repeats (cutoff/error)")

        print("-- primer (Primer-B) --")
        pr_times = timed_primer(spec.path, warmup=N_WARMUP, repeats=N_REPEATS)
        for i, t in enumerate(pr_times, start=1):
            print(f"  primer repeat {i}: {t:.6f}s")
        print(f"  {summarize(pr_times)}")

        if pb_times and pr_times:
            speedup = statistics.mean(pb_times) / statistics.mean(pr_times)
            print(f"  speedup (pyboolnet/primer mean): {speedup:.2f}x")

        if pb_primes is not None and not pb_timeout:
            print("-- equality (canonized) --")
            primer_primes = canon_primes(bnet_file_to_primes(spec.path))
            if primer_primes == pb_primes:
                print(f"  OK  nodes={len(primer_primes)}")
            else:
                diffs = [
                    k
                    for k in sorted(set(primer_primes) | set(pb_primes))
                    if primer_primes.get(k) != pb_primes.get(k)
                ]
                print(
                    f"  FAIL  differing nodes={len(diffs)}  sample={diffs[:10]}"
                )
        else:
            print("-- equality --")
            print("  skipped (PyBoolNet arm did not complete under cutoff)")

        print()

    print(f"finished (UTC): {datetime.now(timezone.utc).isoformat()}")
    sys.stdout = sys.__stdout__
    log_file.close()


if __name__ == "__main__":
    main()
