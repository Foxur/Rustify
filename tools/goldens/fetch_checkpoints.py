"""W1.2: fetch the pinned Laya checkpoints and record what they really are (ADR-005).

    uv run python fetch_checkpoints.py                 # all entries of registry.toml
    uv run python fetch_checkpoints.py --only laya-en
    uv run python fetch_checkpoints.py --local laya-en=/path/to/dir   # inspect an existing dir

For every checkpoint it records, per file, size and sha256, and from the safetensors header the
tensor count, dtypes and parameter count; from rl_agent_config.json the settings the Rust port
depends on (amp_dtype, max_len, head_max_len, temperatures). Answers two open design questions:
  * CR G1: on-disk dtype of model.safetensors (expected F16)
  * CR §3 #9: is the EN root identical at c5d78730 and 1c5edc17?

Output: pins.lock.json next to this script (commit it) and a report under reports/spikes/.
Weights stay in $ARBITRO_HOME/checkpoints (default ~/.cache/arbitro) and are never committed.

NOTE: laya's loader rewrites tokenizer/tokenizer_config.json in place (laya/agent.py
_fix_tokenizer_config). Never point laya at these pristine directories; laya_baseline.py
works on a copy. Re-running this script re-verifies every sha256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import struct
import sys
import tomllib
from collections import Counter
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RUNTIME_FILES = ("rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*")


def arbitro_home() -> Path:
    return Path(os.environ.get("ARBITRO_HOME", Path.home() / ".cache" / "arbitro"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def safetensors_header(path: Path) -> dict:
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    meta = header.pop("__metadata__", None)
    dtypes = Counter(v["dtype"] for v in header.values())
    numel = 0
    for v in header.values():
        k = 1
        for s in v["shape"]:
            k *= s
        numel += k
    prefixes = Counter(name.split(".")[0] for name in header)
    return {
        "tensors": len(header),
        "dtypes": dict(dtypes),
        "elements": numel,
        "prefixes": dict(prefixes),
        "metadata": meta,
        "has_temperature_buffer": "temperature" in header,
    }


def describe_dir(d: Path) -> dict:
    files = {}
    for p in sorted(d.rglob("*")):
        if p.is_file() and ".cache" not in p.parts:
            files[str(p.relative_to(d))] = {"bytes": p.stat().st_size, "sha256": sha256(p)}
    out: dict = {"files": files}
    st = d / "model.safetensors"
    if st.exists():
        out["safetensors"] = safetensors_header(st)
    cfg = d / "rl_agent_config.json"
    if cfg.exists():
        c = json.loads(cfg.read_text())
        keep = ("encoder", "head_layers", "amp_dtype", "max_len", "head_max_len", "act_costs", "cost_wrong_act",
                "temperature", "temperature_by_options")
        out["rl_agent_config"] = {k: c.get(k) for k in keep}
        out["rl_agent_config"]["training"] = c.get("training")
    tc = d / "tokenizer" / "tokenizer_config.json"
    if tc.exists():
        t = json.loads(tc.read_text())
        out["tokenizer_config"] = {k: t.get(k) for k in ("tokenizer_class", "cls_token", "sep_token", "pad_token", "mask_token", "model_max_length")}
    enc = d / "encoder" / "config.json"
    if enc.exists():
        e = json.loads(enc.read_text())
        out["encoder_config"] = {k: e.get(k) for k in ("hidden_size", "num_hidden_layers", "num_attention_heads", "intermediate_size",
                                                       "vocab_size", "local_attention", "layer_types", "rope_parameters",
                                                       "global_rope_theta", "local_rope_theta", "global_attn_every_n_layers")}
    return out


def fetch(entry_id: str, e: dict) -> Path:
    from huggingface_hub import snapshot_download

    sub = e.get("subfolder") or ""
    prefix = f"{sub}/" if sub else ""
    target = arbitro_home() / "checkpoints" / entry_id / e["revision"][:8]
    snapshot_download(
        e["repo"], revision=e["revision"], local_dir=target,
        allow_patterns=[prefix + f for f in RUNTIME_FILES], token=os.environ.get("HF_TOKEN"),
    )
    return target / sub if sub else target


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", action="append", default=[])
    ap.add_argument("--local", action="append", default=[], help="id=DIR: inspect an existing checkpoint dir instead of downloading")
    ap.add_argument("--registry", default=str(HERE / "registry.toml"))
    ap.add_argument("--lock", default=str(HERE / "pins.lock.json"))
    ap.add_argument("--report-dir", default=str(REPO / "reports" / "spikes"))
    args = ap.parse_args(argv)

    registry = tomllib.loads(Path(args.registry).read_text())
    local = dict(x.split("=", 1) for x in args.local)
    ids = args.only or (list(local) if local else list(registry))
    lock = json.loads(Path(args.lock).read_text()) if Path(args.lock).exists() else {}
    results = {}
    for cid in ids:
        e = registry.get(cid, {"repo": "local", "revision": "local"})
        print(f"== {cid} ({e['repo']} @ {e['revision'][:8]})", flush=True)
        d = Path(local[cid]) if cid in local else fetch(cid, e)
        info = describe_dir(d)
        info.update({"repo": e["repo"], "revision": e["revision"], "subfolder": e.get("subfolder", ""), "path": str(d),
                     "compare_only": bool(e.get("compare_only"))})
        prev = lock.get(cid)
        if prev and prev.get("revision") == e["revision"]:
            changed = [f for f, v in info["files"].items() if prev["files"].get(f, {}).get("sha256") != v["sha256"]]
            info["changed_since_lock"] = changed
            if changed:
                print(f"   WARNING: files differ from pins.lock.json: {changed}", flush=True)
        results[cid] = info
        st = info.get("safetensors", {})
        print(f"   tensors={st.get('tensors')} dtypes={st.get('dtypes')} elements={st.get('elements')} "
              f"amp_dtype={info.get('rl_agent_config', {}).get('amp_dtype')}", flush=True)

    findings = []
    for cid, info in results.items():
        st = info.get("safetensors")
        if st:
            findings.append(f"- `{cid}`: {st['tensors']} tensors, dtypes {st['dtypes']}, {st['elements']:,} elements, "
                            f"amp_dtype `{info.get('rl_agent_config', {}).get('amp_dtype')}`, "
                            f"max_len/head_max_len {info.get('rl_agent_config', {}).get('max_len')}/"
                            f"{info.get('rl_agent_config', {}).get('head_max_len')}")
    if "laya-en" in results and "laya-en-at-1c5edc17" in results:
        a = results["laya-en"]["files"].get("model.safetensors", {}).get("sha256")
        b = results["laya-en-at-1c5edc17"]["files"].get("model.safetensors", {}).get("sha256")
        findings.append(f"- EN root weights identical at c5d78730 and 1c5edc17 (CR §3 #9): **{'yes' if a == b else 'NO'}**")

    # Only downloaded registry entries go into the lock, without machine-specific paths.
    lock.update({k: {kk: vv for kk, vv in v.items() if kk not in ("path", "changed_since_lock")}
                 for k, v in results.items() if k not in local})
    Path(args.lock).write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    rep = Path(args.report_dir)
    rep.mkdir(parents=True, exist_ok=True)
    name = f"checkpoints-{socket.gethostname()}-{datetime.now():%Y%m%d-%H%M%S}.md"
    (rep / name).write_text("# Pinned Laya checkpoints\n\n" + "\n".join(findings) + "\n\nPer-file sha256: `tools/goldens/pins.lock.json`.\n")
    print("\n".join(findings))
    print(f"wrote {args.lock}\nwrote {rep / name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
