"""M0 CUDA microbenchmarks (W1.3, via PyTorch): GEMM rates and windowed varlen FlashAttention.

    python -m arbitro_train.bench.gemm [--quick]

Answers, with numbers, the questions the plan left ESTIMATED (docs/ROADMAP.md W1.3, ARCHITECTURE O-1):
  * bf16 and fp16 (fp32 accumulate) GEMM TFLOP/s at the encoder's (N, K) shapes, M in {256..16k}
  * bf16 inputs -> fp32 output (torch.mm out_dtype) exists and what it costs  [fused residual add]
  * FP8 e4m3 torch._scaled_mm with per-tensor and row-wise scales on this GPU  [yes/no + TFLOP/s]
  * INT8 torch._int_mm  [yes/no + TOP/s, informational]
  * varlen FlashAttention-2, head dim 64: window (64,64) vs full, forward and forward+backward
The Rust engine uses cuBLASLt and vendored FA2 directly; these PyTorch numbers are the ceiling
reference that the engine spike (M4) is compared with.
"""

from __future__ import annotations

import argparse
import sys

import torch

from .common import cuda_time, host_info, md_table, write_report

SHAPES = {
    "modernbert-large": {"Wqkv": (3072, 1024), "attn.Wo": (1024, 1024), "mlp.Wi": (5248, 1024), "mlp.Wo": (1024, 2624)},
    "modernbert-base": {"Wqkv": (2304, 768), "attn.Wo": (768, 768), "mlp.Wi": (2304, 768), "mlp.Wo": (768, 1152)},
}


def _tflops(m, n, k, ms):
    return 2.0 * m * n * k / (ms * 1e-3) / 1e12


def bench_gemm(ms_list, shapes) -> list[dict]:
    rows = []
    torch.backends.cuda.matmul.allow_tf32 = False
    for model, layers in shapes.items():
        for lname, (n, k) in layers.items():
            w16 = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
            for m in ms_list:
                a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
                row = {"model": model, "layer": lname, "M": m, "N": n, "K": k}
                row["bf16"] = _tflops(m, n, k, cuda_time(lambda: a @ w16.t()))
                a16, w_16 = a.half(), w16.half()
                torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
                row["fp16_acc32"] = _tflops(m, n, k, cuda_time(lambda: a16 @ w_16.t()))
                torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
                try:
                    wt = w16.t()
                    row["bf16_to_fp32_out"] = _tflops(m, n, k, cuda_time(lambda: torch.mm(a, wt, out_dtype=torch.float32)))
                except Exception as e:
                    row["bf16_to_fp32_out"] = f"n/a ({type(e).__name__})"
                row.update(_fp8(a, w16, m, n, k))
                row.update(_int8(m, n, k))
                rows.append(row)
                print(row, flush=True)
    return rows


def _fp8(a, w, m, n, k) -> dict:
    out = {}
    if not hasattr(torch, "float8_e4m3fn"):
        return {"fp8_tensor": "n/a", "fp8_rowwise": "n/a"}
    a8 = a.to(torch.float8_e4m3fn)
    b8 = w.to(torch.float8_e4m3fn).t()  # column-major [K, N]
    one = torch.tensor(1.0, device="cuda")
    for fast in (False, True):
        key = "fp8_tensor" + ("_fastacc" if fast else "")
        try:
            out[key] = _tflops(m, n, k, cuda_time(lambda: torch._scaled_mm(a8, b8, one, one, out_dtype=torch.bfloat16, use_fast_accum=fast)))
        except Exception as e:
            out[key] = f"n/a ({type(e).__name__}: {str(e)[:80]})"
    try:
        sa = torch.ones(m, 1, device="cuda")
        sb = torch.ones(1, n, device="cuda")
        out["fp8_rowwise"] = _tflops(m, n, k, cuda_time(lambda: torch._scaled_mm(a8, b8, sa, sb, out_dtype=torch.bfloat16)))
    except Exception as e:
        out["fp8_rowwise"] = f"n/a ({type(e).__name__}: {str(e)[:80]})"
    return out


def _int8(m, n, k) -> dict:
    try:
        a = torch.randint(-127, 127, (m, k), device="cuda", dtype=torch.int8)
        b = torch.randint(-127, 127, (k, n), device="cuda", dtype=torch.int8)
        return {"int8": _tflops(m, n, k, cuda_time(lambda: torch._int_mm(a, b)))}
    except Exception as e:
        return {"int8": f"n/a ({type(e).__name__})"}


def bench_flash(total_tokens: int, seq_len: int, heads: int = 16, head_dim: int = 64) -> list[dict]:
    from torch.nn.attention.varlen import varlen_attn

    n = total_tokens // seq_len
    t = n * seq_len
    cu = torch.arange(0, t + 1, seq_len, device="cuda", dtype=torch.int32)
    rows = []
    for name, ws in (("full", (-1, -1)), ("window64", (64, 64))):
        q = torch.randn(t, heads, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn_like(q, requires_grad=True)
        v = torch.randn_like(q, requires_grad=True)
        if ws[0] < 0:
            pairs = n * seq_len * seq_len
        else:
            i = torch.arange(seq_len)
            pairs = n * int(((i[:, None] - i[None, :]).abs() <= ws[0]).sum())
        flops_fwd = 4.0 * pairs * heads * head_dim
        fwd = cuda_time(lambda: varlen_attn(q, k, v, cu, cu, seq_len, seq_len, window_size=ws))

        def fb():
            o = varlen_attn(q, k, v, cu, cu, seq_len, seq_len, window_size=ws)
            o.backward(torch.ones_like(o))

        fwdbwd = cuda_time(fb)
        rows.append({
            "attn": name, "tokens": t, "seq_len": seq_len, "heads": heads, "head_dim": head_dim,
            "fwd_ms": fwd, "fwd_tflops": flops_fwd / (fwd * 1e-3) / 1e12,
            "fwd_bwd_ms": fwdbwd, "fwd_bwd_tflops": 3.5 * flops_fwd / (fwdbwd * 1e-3) / 1e12,
        })
        print(rows[-1], flush=True)
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="fewer shapes (about 1 minute)")
    ap.add_argument("--out-dir")
    args = ap.parse_args(argv)
    if not torch.cuda.is_available():
        print("CUDA not available")
        return 1
    ms_list = [1024, 12288] if args.quick else [256, 1024, 4096, 12288, 16384]
    shapes = {"modernbert-large": SHAPES["modernbert-large"]} if args.quick else SHAPES
    gemm = bench_gemm(ms_list, shapes)
    flash = []
    for L in ([512] if args.quick else [128, 512, 1024]):
        try:
            flash += bench_flash(16384, L)
        except Exception as e:
            flash.append({"attn": "error", "seq_len": L, "error": f"{type(e).__name__}: {e}"})

    cols = ["model", "layer", "M", "N", "K", "bf16", "fp16_acc32", "bf16_to_fp32_out", "fp8_tensor", "fp8_tensor_fastacc", "fp8_rowwise", "int8"]
    md = "# CUDA microbenchmarks (PyTorch)\n\nTFLOP/s (TOP/s for int8); `n/a` means the op is not supported here.\n\n"
    md += md_table(gemm, cols)
    md += "\n## varlen FlashAttention-2 (bf16, head dim 64)\n\n"
    md += md_table(flash, ["attn", "tokens", "seq_len", "fwd_ms", "fwd_tflops", "fwd_bwd_ms", "fwd_bwd_tflops"])
    write_report("cuda-gemm", {"host": host_info(), "gemm": gemm, "flash": flash}, md, args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
