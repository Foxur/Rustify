"""Summarise E1 runs into one Markdown report (reports/e1/).

    python -m arbitro_train.report_e1                 # all runs under <repo>/runs
    python -m arbitro_train.report_e1 runs/<a> runs/<b>
"""

from __future__ import annotations

import json
import socket
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    runs = [Path(a) for a in argv] or sorted(p.parent for p in (REPO / "runs").glob("*/summary.json"))
    if not runs:
        print("no finished runs (runs/*/summary.json)")
        return 1
    lines = ["# E1 early signal\n",
             "Signals, not model results (docs/TRAINING.md §5.3). An arm *learns* when the 95 % bootstrap CI lower bound "
             "of its OOD-S dev accuracy is above chance.\n",
             "| run | backbone | tokens | OOD-S set | acc | 95 % CI | chance | ECE-15 | learns |", "|---|---|---|---|---|---|---|---|---|"]
    curves = []
    for r in runs:
        s = json.loads((r / "summary.json").read_text())
        cfg = json.loads((r / "config.json").read_text())
        bb = cfg["model"]["backbone"]
        for k, v in s["e1_signal"].items():
            ece = s["final"][k]["ece15"]
            lines.append(f"| `{r.name}` | {bb} | {s['tokens']:,} | {k} | {v['acc']:.3f} | [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}] | "
                         f"{v['chance']:.3f} | {ece:.3f} | {'yes' if v['learns'] else 'no'} |")
        evals = [json.loads(line) for line in open(r / "metrics.jsonl") if '"kind": "eval"' in line]
        if evals:
            sets = list(evals[-1]["sets"])
            curves.append(f"\n### `{r.name}` ({bb}) learning curve\n")
            curves.append("| tokens | " + " | ".join(sets) + " |")
            curves.append("|---|" + "---|" * len(sets))
            for e in evals:
                curves.append(f"| {e['tokens']:,} | " + " | ".join(f"{e['sets'][k]['acc']:.3f}" for k in sets) + " |")
    out = REPO / "reports" / "e1"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"e1-{socket.gethostname()}-{datetime.now():%Y%m%d-%H%M%S}.md"
    path.write_text("\n".join(lines + curves) + "\n")
    print("\n".join(lines))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
