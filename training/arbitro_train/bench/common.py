"""Shared helpers for the M0 spikes: host facts, CUDA timing, report files."""

from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import shutil
import socket
import subprocess
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REPORT_DIR = REPO_ROOT / "reports" / "spikes"


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ""


def nvidia_smi() -> dict:
    fields = [
        "name", "driver_version", "memory.total", "power.limit", "power.max_limit",
        "clocks.max.sm", "clocks.max.mem", "display_active", "persistence_mode", "pcie.link.gen.max",
        "pcie.link.width.max", "temperature.gpu",
    ]
    out = _run(["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"])
    if not out:
        return {}
    vals = [v.strip() for v in out.splitlines()[0].split(",")]
    return dict(zip(fields, vals))


def host_info() -> dict:
    info = {
        "hostname": socket.gethostname(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "cpu": _run(["sh", "-c", "grep -m1 'model name' /proc/cpuinfo | cut -d: -f2"]).strip() or platform.processor(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import transformers

        info["transformers"] = transformers.__version__
    except Exception:
        pass
    try:
        info["triton"] = __import__("triton").__version__
    except Exception:
        info["triton"] = None
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update({
            "gpu": p.name,
            "capability": f"{p.major}.{p.minor}",
            "gpu_mem_gib": round(p.total_memory / 2**30, 2),
            "sm_count": p.multi_processor_count,
        })
        info["nvidia_smi"] = nvidia_smi()
    try:
        mem = _run(["sh", "-c", "grep MemTotal /proc/meminfo"]).split()
        info["ram_gib"] = round(int(mem[1]) / 2**20, 1)
    except Exception:
        pass
    du = shutil.disk_usage(REPO_ROOT)
    info["disk_free_gib"] = round(du.free / 2**30, 1)
    return info


def cuda_time(fn, warmup: int = 3, iters: int = 10) -> float:
    """Median milliseconds of ``fn()`` measured with CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2]


def stamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def write_report(kind: str, payload: dict, markdown: str, out_dir: str | Path | None = None) -> tuple[Path, Path]:
    """Write ``<kind>-<host>-<stamp>.json`` and ``.md`` under reports/spikes/."""
    d = Path(out_dir) if out_dir else DEFAULT_REPORT_DIR
    d.mkdir(parents=True, exist_ok=True)
    base = f"{kind}-{socket.gethostname()}-{stamp()}"
    jp, mp = d / f"{base}.json", d / f"{base}.md"
    jp.write_text(json.dumps(payload, indent=2, default=str))
    mp.write_text(markdown)
    print(f"wrote {jp}\nwrote {mp}", flush=True)
    return jp, mp


def md_table(rows: list[dict], cols: list[str]) -> str:
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    body = ""
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            cells.append(f"{v:,.1f}" if isinstance(v, float) else str(v))
        body += "| " + " | ".join(cells) + " |\n"
    return head + body
