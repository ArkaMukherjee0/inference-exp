"""Environment provenance capture.

Called once per condition-visit and merged into every record it produces. If a number
in the report cannot be traced back to the exact stack version, driver and clock state
that produced it, it does not belong in the report.

Nothing here guesses. Where a value cannot be read it is ``None`` and the caller
decides whether that is fatal -- except on a GPU platform, where a missing nvidia-smi
is fatal immediately, because it means we cannot verify clocks or exclusivity.
"""

from __future__ import annotations

import os
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from typing import Any

_NVIDIA_SMI_TIMEOUT_S = 15


def utc_now() -> str:
    """ISO-8601 UTC. One timezone across three cloud instances and a desktop."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _run(cmd: list[str], *, timeout: int = _NVIDIA_SMI_TIMEOUT_S) -> str | None:
    exe = shutil.which(cmd[0])
    if exe is None:
        return None
    try:
        out = subprocess.run(
            [exe, *cmd[1:]], capture_output=True, text=True, timeout=timeout, check=True
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return None
    return out.stdout.strip()


# --------------------------------------------------------------------------------------
# GPU
# --------------------------------------------------------------------------------------


def _smi_query(fields: list[str]) -> list[list[str]] | None:
    raw = _run(["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"])
    if raw is None:
        return None
    return [[c.strip() for c in line.split(",")] for line in raw.splitlines() if line.strip()]


def _as_float(value: str) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    # nvidia-smi reports unsupported metrics as [N/A] -> None above, and 0 for a GPU
    # that is genuinely idle. Zero clocks would be a lie in a record, so reject them.
    return f if f > 0 else None


def gpu_sample() -> dict[str, Any]:
    """Per-record GPU state: SM clock and power draw, sampled at measurement time."""
    rows = _smi_query(["clocks.sm", "power.draw"])
    if not rows:
        return {"clocks_sm_mhz": None, "power_draw_w": None}
    # Device 0 is the one we pin to; TP runs report device 0 as representative and the
    # full per-device list lives in the env blob.
    return {
        "clocks_sm_mhz": _as_float(rows[0][0]),
        "power_draw_w": _as_float(rows[0][1]) if len(rows[0]) > 1 else None,
    }


def gpu_info() -> dict[str, Any] | None:
    rows = _smi_query(["name", "driver_version", "memory.total", "clocks.max.sm", "persistence_mode"])
    if not rows:
        return None
    return {
        "devices": [r[0] for r in rows],
        "count": len(rows),
        "driver_version": rows[0][1],
        "memory_total_mib": rows[0][2],
        "clocks_max_sm_mhz": rows[0][3],
        "persistence_mode": rows[0][4] if len(rows[0]) > 4 else None,
    }


def gpu_resident_processes() -> list[dict[str, str]] | None:
    """Every process holding GPU memory, ours included.

    A neighbouring tenant is the difference between a measurement and a story about
    contention, so the driver re-checks this before each condition block.
    """
    raw = _run([
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ])
    if raw is None:
        return None
    procs = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        parts = [c.strip() for c in line.split(",")]
        if len(parts) >= 3:
            procs.append({"pid": parts[0], "name": parts[1], "used_mib": parts[2]})
    return procs


def assert_gpu_exclusive() -> None:
    """Raise unless this process is the only one resident on the GPU."""
    procs = gpu_resident_processes()
    if procs is None:
        raise RuntimeError(
            "GPU exclusivity could not be verified: nvidia-smi is unavailable. Refusing "
            "to measure on a GPU whose occupancy is unknown."
        )
    mine = str(os.getpid())
    foreign = [p for p in procs if p["pid"] != mine]
    if foreign:
        raise RuntimeError(
            "GPU is not exclusive; the following processes hold memory and will "
            f"contend for bandwidth: {foreign}. Stop them and re-run -- do not measure "
            "through contention."
        )


# --------------------------------------------------------------------------------------
# CPU
# --------------------------------------------------------------------------------------


def cpu_model() -> str:
    """Best available CPU identification, per OS. Never a hardcoded guess."""
    if sys.platform == "win32":
        out = _run(["wmic", "cpu", "get", "name"], timeout=20)
        if out:
            lines = [ln.strip() for ln in out.splitlines() if ln.strip() and "Name" not in ln]
            if lines:
                return lines[0]
        ps = _run([
            "powershell", "-NoProfile", "-Command",
            "(Get-CimInstance Win32_Processor).Name",
        ], timeout=30)
        if ps:
            return ps.splitlines()[0].strip()
    elif sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    elif sys.platform == "darwin":
        out = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if out:
            return out
    return platform.processor() or platform.machine() or "unknown"


def cpu_info() -> dict[str, Any]:
    return {
        "model": cpu_model(),
        "logical_cores": os.cpu_count(),
        "affinity_cores": _affinity_count(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
    }


def _affinity_count() -> int | None:
    """How many cores this process may actually run on.

    On a hybrid P/E-core desktop the affinity mask is the difference between a
    reproducible timing and a wandering one, so it is recorded, not assumed.
    """
    fn = getattr(os, "sched_getaffinity", None)
    if fn is not None:
        try:
            return len(fn(0))
        except OSError:
            return None
    return None


# --------------------------------------------------------------------------------------
# Software versions
# --------------------------------------------------------------------------------------


def _dist_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def stack_version(stack: str, *, llamacpp_binary: str | None = None) -> str:
    """The version string of the engine that produced a measurement.

    Raises rather than returning "unknown": a record whose stack version is unknown
    cannot be compared with anything, so it must not be written.
    """
    if stack == "vllm":
        v = _dist_version("vllm")
        if v is None:
            raise RuntimeError("stack='vllm' but vLLM is not installed in this interpreter.")
        return f"vllm=={v}"
    if stack == "hf":
        v = _dist_version("transformers")
        if v is None:
            raise RuntimeError("stack='hf' but transformers is not installed in this interpreter.")
        return f"transformers=={v}"
    if stack == "llamacpp":
        if not llamacpp_binary:
            raise RuntimeError("stack='llamacpp' requires the binary path to read its version.")
        out = _run([llamacpp_binary, "--version"], timeout=30)
        if not out:
            raise RuntimeError(
                f"could not read a version from {llamacpp_binary!r}. llama.cpp builds vary; "
                "refusing to record a measurement from an unidentifiable binary."
            )
        return f"llamacpp:{out.splitlines()[0].strip()}"
    raise ValueError(f"unknown stack {stack!r}")


def driver_string(platform_name: str) -> str:
    """CUDA driver on GPU platforms; OS/kernel identification on CPU."""
    if platform_name == "cpu":
        return f"{platform.system()} {platform.release()}"
    info = gpu_info()
    if info is None:
        raise RuntimeError(
            f"platform={platform_name!r} but nvidia-smi is unavailable. Refusing to record "
            "a GPU measurement with no driver provenance."
        )
    cuda = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    return f"driver={info['driver_version']}" + (f" smi={cuda.splitlines()[0]}" if cuda else "")


# --------------------------------------------------------------------------------------
# Top-level capture
# --------------------------------------------------------------------------------------


def capture_env(
    *,
    stack: str,
    platform_name: str,
    llamacpp_binary: str | None = None,
    require_exclusive: bool = True,
) -> dict[str, Any]:
    """Full provenance blob, called once per condition-visit.

    Returns a dict carrying the two flat fields every record needs
    (``hostname``, ``stack_version``, ``driver``) plus a nested ``env`` blob with
    everything else.
    """
    if platform_name != "cpu" and require_exclusive:
        assert_gpu_exclusive()

    versions = {
        name: _dist_version(name)
        for name in ("vllm", "transformers", "torch", "numpy", "pandas", "matplotlib", "seaborn")
    }
    torch_env: dict[str, Any] = {}
    try:  # torch is not installed on a plotting-only host, and that is fine
        import torch  # noqa: PLC0415

        torch_env = {
            "torch_version": torch.__version__,
            "cuda_version": getattr(torch.version, "cuda", None),
            "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            "cuda_available": torch.cuda.is_available(),
        }
    except ImportError:
        torch_env = {"torch_version": None}

    return {
        "hostname": socket.gethostname(),
        "stack_version": stack_version(stack, llamacpp_binary=llamacpp_binary),
        "driver": driver_string(platform_name),
        "env": {
            # Engine env vars change which kernels run, so they are provenance, not
            # configuration trivia. VLLM_USE_FLASHINFER_SAMPLER in particular decides
            # whether sampling goes through a JIT-compiled FlashInfer kernel or the
            # native path -- a difference in per-step overhead that would otherwise vary
            # silently between boxes depending on whether a CUDA toolkit is installed.
            "engine_env": {
                k: v for k, v in os.environ.items()
                if k.startswith(("VLLM_", "NCCL_", "CUDA_VISIBLE", "TORCH_"))
            },
            "captured_at": utc_now(),
            "python": sys.version.split()[0],
            "os": f"{platform.system()} {platform.release()} ({platform.version()})",
            "cpu": cpu_info(),
            "gpu": gpu_info(),
            "gpu_resident_processes": gpu_resident_processes(),
            "packages": versions,
            **torch_env,
        },
    }
