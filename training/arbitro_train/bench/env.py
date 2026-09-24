"""M0 bring-up check for the training machine (W1.2 prerequisites, Q6).

    python -m arbitro_train.bench.env

Prints a readiness verdict and writes reports/spikes/env-<host>-<stamp>.{json,md}.
"""

from __future__ import annotations

import argparse
import platform
import sys

import torch

from .common import host_info, write_report


def _check_varlen() -> tuple[bool, str]:
    try:
        from torch.nn.attention.varlen import varlen_attn
    except Exception as e:  # pragma: no cover
        return False, f"import failed: {e}"
    try:
        q = torch.randn(96, 4, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn_like(q, requires_grad=True)
        v = torch.randn_like(q, requires_grad=True)
        cu = torch.tensor([0, 40, 96, 96], device="cuda", dtype=torch.int32)  # incl. a zero-length segment
        out = varlen_attn(q, k, v, cu, cu, 64, 64, window_size=(8, 8))
        out.float().sum().backward()
        torch.cuda.synchronize()
        return bool(torch.isfinite(out).all() and torch.isfinite(q.grad).all()), "fwd+bwd with window (8,8) and a zero-length segment"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _check_compile() -> tuple[bool, str]:
    try:
        f = torch.compile(lambda x: torch.nn.functional.gelu(x) * 2)
        y = f(torch.randn(128, device="cuda"))
        torch.cuda.synchronize()
        return bool(torch.isfinite(y).all()), "inductor on CUDA"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:300]}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir")
    args = ap.parse_args(argv)

    info = host_info()
    checks: list[dict] = []

    def add(name: str, ok: bool | None, detail: str, blocking: bool = True):
        checks.append({"check": name, "ok": ok, "detail": detail, "blocking": blocking})

    is_linux = platform.system() == "Linux"
    add("os_linux", is_linux, info["os"] + ("" if is_linux else " - use Linux or WSL2: varlen FlashAttention and torch.compile/triton need it"))
    cuda = torch.cuda.is_available()
    add("cuda_available", cuda, f"torch {torch.__version__}, CUDA runtime {torch.version.cuda}")
    if cuda:
        major, minor = torch.cuda.get_device_capability(0)
        add("compute_capability>=8.0", major >= 8, f"sm_{major}{minor} ({info.get('gpu')})")
        add("bf16_supported", torch.cuda.is_bf16_supported(), "bf16 autocast")
        smi = info.get("nvidia_smi", {})
        drv = smi.get("driver_version", "")
        try:
            drv_major = int(drv.split(".")[0])
        except ValueError:
            drv_major = 0
        need = 580 if (torch.version.cuda or "").startswith("13") else 525
        add("driver_version", drv_major >= need, f"driver {drv or 'unknown'} (need >= {need} for CUDA {torch.version.cuda} wheels)")
        disp = smi.get("display_active", "")
        add("headless", disp.lower() != "enabled", f"display_active={disp or 'unknown'}: a desktop costs 0.3-1 GB VRAM and adds timing noise", blocking=False)
        add("power_limit_recorded", True, f"power.limit={smi.get('power.limit')} W (max {smi.get('power.max_limit')} W); training plan assumes 350-380 W (C6)", blocking=False)
        free, total = torch.cuda.mem_get_info()
        add("vram_free", free / 2**30 > 20, f"{free / 2**30:.1f} of {total / 2**30:.1f} GiB free")
        ok, det = _check_varlen()
        add("varlen_attn_fwd_bwd", ok, det)
        ok, det = _check_compile()
        add("torch_compile", ok, det, blocking=False)
    add("disk_free", info.get("disk_free_gib", 0) >= 200, f"{info.get('disk_free_gib')} GiB free on the repo volume (>= 1 TB recommended for M3a/M5, Q6)", blocking=False)
    add("ram", info.get("ram_gib", 0) >= 32, f"{info.get('ram_gib')} GiB RAM", blocking=False)

    ready = all(c["ok"] for c in checks if c["blocking"])
    for c in checks:
        flag = "OK  " if c["ok"] else ("FAIL" if c["blocking"] else "WARN")
        print(f"[{flag}] {c['check']}: {c['detail']}")
    print("READY for M0 GPU spikes" if ready else "NOT READY - fix the FAIL lines first")

    md = ["# Environment check\n", f"Verdict: **{'ready' if ready else 'not ready'}**\n", "| Check | Result | Detail |", "|---|---|---|"]
    for c in checks:
        md.append(f"| {c['check']} | {'ok' if c['ok'] else ('FAIL' if c['blocking'] else 'warn')} | {c['detail']} |")
    md.append("\n```json\n" + __import__("json").dumps(info, indent=2, default=str) + "\n```\n")
    write_report("env", {"ready": ready, "checks": checks, "host": info}, "\n".join(md), args.out_dir)
    return 0 if ready else 1


if __name__ == "__main__":
    sys.exit(main())
