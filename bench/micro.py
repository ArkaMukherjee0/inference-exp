"""Microbenchmarks: achieved memory bandwidth and achieved compute throughput.

Figure 05 places platforms along a ridge-point axis, so the ridge point had better be a
property of the machine rather than of its marketing. Nothing in this module reads a
spec-sheet constant -- there is no table of peak TFLOPS, no hardcoded HBM bandwidth, and
a test asserts as much. Every number is measured here and now, on this machine, and
recorded with the size and repeat count that produced it.

Peak-to-achieved ratios differ substantially between an H100 and a desktop CPU. Using
advertised figures would compress or stretch the very axis figure 05 is built on, in a
direction that depends on which vendor was more optimistic.
"""

from __future__ import annotations

import json
import os
import platform
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from analysis.model import ridge_point
from core.env import capture_env, cpu_info, gpu_info


@dataclass
class BandwidthResult:
    gbytes_per_s: float
    array_bytes: int
    n_elements: int
    dtype: str
    repeats: int
    best_seconds: float
    all_seconds: list[float] = field(default_factory=list)
    threads: int = 1
    kernel: str = "stream_add"


@dataclass
class ComputeResult:
    tflops: float
    m: int
    n: int
    k: int
    dtype: str
    repeats: int
    best_seconds: float
    all_seconds: list[float] = field(default_factory=list)


@dataclass
class MicroReport:
    platform: str
    device: str
    bandwidth: dict[str, Any]
    compute: dict[str, Any]
    ridge_point_flop_per_byte: float
    env: dict[str, Any]

    def write(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2, default=str), encoding="utf-8")
        return p


# --------------------------------------------------------------------------------------
# CPU
# --------------------------------------------------------------------------------------


def cpu_bandwidth(
    *,
    n_elements: int = 1 << 26,
    repeats: int = 10,
    dtype: str = "float64",
    threads: int | None = None,
) -> BandwidthResult:
    """STREAM **Add** kernel, ``a = b + c``, run across a thread pool.

    Exactly three arrays are touched per element -- two read, one written -- so traffic is
    ``3 * n * itemsize`` bytes. Arrays are sized well past last-level cache, so this
    measures memory rather than cache.

    Two things here are load-bearing, and both were wrong in an earlier version in ways
    that understated the result by roughly an order of magnitude:

    **No temporary inside the timed region.** The textbook triad ``a = b + scalar*c`` has
    no fused NumPy form: written naively, ``scalar * c`` allocates a fresh array the size
    of the input on *every iteration*, so the loop measures page-fault throughput on a
    half-gigabyte allocation rather than memory bandwidth -- and the real traffic is five
    arrays, not the three the result is divided by. The Add kernel is a single ufunc call
    with ``out=``, allocates nothing, and its traffic is exactly countable. It is a
    standard STREAM kernel in its own right; we use it instead of Triad precisely because
    it maps to one allocation-free call.

    **Threads.** NumPy ufuncs are single-threaded, so a one-core run measures that core's
    share of the memory system, not the machine's. One core cannot saturate a modern
    multi-channel controller. The array is split into contiguous slices and each thread
    runs the kernel over its own slice; NumPy releases the GIL inside ufunc inner loops,
    so this is genuinely parallel. The thread count is recorded, because a bandwidth
    figure without one is not reproducible.

    We report the *best* of several repeats: for a ceiling, the fastest clean run is the
    least contaminated by scheduler interference, whereas the mean partly measures the
    machine's background noise. Every individual timing is kept so that choice is
    auditable.
    """
    if threads is None:
        # Sweep and keep the best, exactly as the GEMM sweep keeps the best size. Memory
        # bandwidth saturates well below the core count and then *falls* as extra threads
        # oversubscribe the controller, so "all cores" is not the achieved ceiling -- on
        # this desktop it is about 10% below it.
        return _best_over_threads(
            n_elements=n_elements, repeats=repeats, dtype=dtype
        )

    dt = np.dtype(dtype)
    n_threads = max(1, min(threads, n_elements))

    a = np.empty(n_elements, dtype=dt)
    b = np.ones(n_elements, dtype=dt)
    c = np.full(n_elements, 2.0, dtype=dt)

    bounds = np.linspace(0, n_elements, n_threads + 1).astype(int)
    slices = [slice(int(bounds[i]), int(bounds[i + 1])) for i in range(n_threads)]

    def _kernel(sl: slice) -> None:
        np.add(b[sl], c[sl], out=a[sl])

    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        # Untimed pass: fault in every page and spin up the workers, so neither a
        # first-touch fault storm nor thread creation is charged to the memory system.
        list(pool.map(_kernel, slices))

        times: list[float] = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            list(pool.map(_kernel, slices))
            times.append(time.perf_counter() - t0)

    best = min(times)
    traffic = 3 * n_elements * dt.itemsize
    return BandwidthResult(
        gbytes_per_s=traffic / best / 1e9,
        array_bytes=n_elements * dt.itemsize,
        n_elements=n_elements,
        dtype=str(dt),
        repeats=repeats,
        best_seconds=best,
        all_seconds=times,
        threads=n_threads,
        kernel="stream_add",
    )


def _best_over_threads(*, n_elements: int, repeats: int, dtype: str) -> BandwidthResult:
    """Run the Add kernel at several thread counts and keep the fastest.

    The winning thread count is recorded on the result, because a bandwidth number
    without one cannot be reproduced or argued with.
    """
    cores = os.cpu_count() or 1
    candidates = sorted({1, 2, 4, 8, 16, cores})
    candidates = [t for t in candidates if t <= max(cores, 1)]

    best: BandwidthResult | None = None
    for t in candidates:
        result = cpu_bandwidth(
            n_elements=n_elements, repeats=repeats, dtype=dtype, threads=t
        )
        if best is None or result.gbytes_per_s > best.gbytes_per_s:
            best = result
    assert best is not None
    return best


def cpu_compute(*, sizes: tuple[int, ...] = (2048, 4096, 6144), repeats: int = 5,
                dtype: str = "float32") -> ComputeResult:
    """Large square GEMM sweep; the best size wins.

    A single size can land badly against cache geometry, so we sweep and take the best
    achieved throughput -- which is what "achieved peak" means. FLOP count for an
    ``m x k`` by ``k x n`` product is ``2*m*n*k`` (one multiply and one add per term).
    """
    dt = np.dtype(dtype)
    best: ComputeResult | None = None
    for size in sizes:
        a = np.random.default_rng(0).random((size, size)).astype(dt)
        b = np.random.default_rng(1).random((size, size)).astype(dt)
        a @ b  # warm up BLAS threads and any lazy allocation
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            a @ b
            times.append(time.perf_counter() - t0)
        t = min(times)
        tflops = (2.0 * size ** 3) / t / 1e12
        if best is None or tflops > best.tflops:
            best = ComputeResult(
                tflops=tflops, m=size, n=size, k=size, dtype=str(dt),
                repeats=repeats, best_seconds=t, all_seconds=times,
            )
    assert best is not None
    return best


# --------------------------------------------------------------------------------------
# GPU
# --------------------------------------------------------------------------------------


def gpu_bandwidth(*, n_elements: int = 1 << 28, repeats: int = 10, dtype: str = "float32") -> BandwidthResult:
    """The same triad on the GPU, timed with CUDA events.

    ``perf_counter`` around an async kernel launch measures the launch, not the kernel,
    so CUDA events are used and the device is synchronized before reading them.
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("gpu_bandwidth called but CUDA is not available.")
    dt = getattr(torch, dtype)
    dev = torch.device("cuda")
    a = torch.zeros(n_elements, dtype=dt, device=dev)
    b = torch.ones(n_elements, dtype=dt, device=dev)
    c = torch.full((n_elements,), 2.0, dtype=dt, device=dev)
    scalar = 3.0

    torch.add(b, c, alpha=scalar, out=a)
    torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        torch.add(b, c, alpha=scalar, out=a)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / 1000.0)

    best = min(times)
    traffic = 3 * n_elements * a.element_size()
    return BandwidthResult(
        gbytes_per_s=traffic / best / 1e9,
        array_bytes=n_elements * a.element_size(),
        n_elements=n_elements,
        dtype=str(dt),
        repeats=repeats,
        best_seconds=best,
        all_seconds=times,
        threads=1,
        kernel="stream_triad_fused",
    )


def gpu_compute(*, sizes: tuple[int, ...] = (4096, 8192, 16384), repeats: int = 10,
                dtype: str = "bfloat16") -> ComputeResult:
    """Large GEMM sweep at the dtype the study actually runs.

    bfloat16 by default because that is the baseline precision of the sweeps; running
    this at fp32 would measure a unit the models never touch.
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("gpu_compute called but CUDA is not available.")
    dt = getattr(torch, dtype)
    dev = torch.device("cuda")
    best: ComputeResult | None = None

    for size in sizes:
        try:
            a = torch.randn((size, size), dtype=dt, device=dev)
            b = torch.randn((size, size), dtype=dt, device=dev)
        except torch.cuda.OutOfMemoryError:
            # Skipping a size that does not fit is honest; the sizes that did run are
            # recorded, and there is no substitution of a made-up number.
            continue
        torch.matmul(a, b)
        torch.cuda.synchronize()

        times = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            torch.matmul(a, b)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end) / 1000.0)

        t = min(times)
        tflops = (2.0 * size ** 3) / t / 1e12
        if best is None or tflops > best.tflops:
            best = ComputeResult(
                tflops=tflops, m=size, n=size, k=size, dtype=str(dt),
                repeats=repeats, best_seconds=t, all_seconds=times,
            )
        del a, b
        torch.cuda.empty_cache()

    if best is None:
        raise RuntimeError(
            f"no GEMM size in {sizes} fitted in GPU memory; cannot report achieved "
            "throughput. Reduce the sizes rather than accepting a missing measurement."
        )
    return best


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------


def run(platform_name: str, *, outdir: str | Path = "logs") -> MicroReport:
    """Measure this machine and write ``logs/{hostname}_micro.json``."""
    if platform_name not in ("cpu", "h100"):
        raise ValueError(f"unknown platform {platform_name!r}")

    if platform_name == "cpu":
        bw = cpu_bandwidth()
        comp = cpu_compute()
        device = cpu_info()["model"]
    else:
        bw = gpu_bandwidth()
        comp = gpu_compute()
        info = gpu_info()
        if info is None:
            raise RuntimeError("platform='h100' but no GPU was visible to nvidia-smi.")
        device = info["devices"][0]

    env = capture_env(
        stack="hf" if platform_name == "cpu" else "vllm",
        platform_name=platform_name,
        require_exclusive=False,
    ) if _stack_available(platform_name) else {"env": {"note": "stack version unavailable"}}

    report = MicroReport(
        platform=platform_name,
        device=device,
        bandwidth=asdict(bw),
        compute=asdict(comp),
        ridge_point_flop_per_byte=ridge_point(comp.tflops, bw.gbytes_per_s),
        env=env.get("env", {}),
    )
    import socket

    report.write(Path(outdir) / f"{socket.gethostname()}_micro.json")
    return report


def _stack_available(platform_name: str) -> bool:
    """Version capture needs the engine installed; the benchmark itself does not."""
    from importlib import metadata

    name = "transformers" if platform_name == "cpu" else "vllm"
    try:
        metadata.version(name)
    except metadata.PackageNotFoundError:
        return False
    return True


def load_report(path: str | Path) -> MicroReport:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return MicroReport(**data)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--platform", required=True, choices=["cpu", "h100"])
    ap.add_argument("--outdir", default="logs")
    args = ap.parse_args()

    rep = run(args.platform, outdir=args.outdir)
    print(f"device: {rep.device} ({platform.system()})")
    print(f"achieved bandwidth: {rep.bandwidth['gbytes_per_s']:.1f} GB/s "
          f"({rep.bandwidth['dtype']}, {rep.bandwidth['n_elements']} elements, "
          f"best of {rep.bandwidth['repeats']})")
    print(f"achieved compute:   {rep.compute['tflops']:.2f} TFLOP/s "
          f"({rep.compute['dtype']}, {rep.compute['m']}^3, best of {rep.compute['repeats']})")
    print(f"ridge point:        {rep.ridge_point_flop_per_byte:.1f} FLOP/byte")
