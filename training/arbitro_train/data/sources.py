"""Fetch and convert the E1 sources into canonical JSONL records (docs/TRAINING.md §5.3, §11).

    python -m arbitro_train.data.sources                  # all E1 sources
    python -m arbitro_train.data.sources --only clinc150

Every source has a manifest in data/manifests/<id>.toml (URL, pinned revision, sha256, SPDX id,
allowed_use). Downloads go to $ARBITRO_HOME/data/raw/<id>/ and are verified against the manifest
sha256; files without a pinned sha256 (PAWS, ANLI) are recorded trust-on-first-use in
data/data.lock.json, which is committed. Converted records go to $ARBITRO_HOME/data/e1/.

Canonical record (one JSON object per line):
    {"id", "source", "split", "task", "state": str | dict, "label": str}          single label
    {"id", "source", "split", "task", "state": str | dict, "labels": [str, ...]}  multi-label
plus <id>.labels.json with the label space. Tasks: intent, emotion, paraphrase, nli.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
import tarfile
import tomllib
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
MANIFESTS = REPO / "data" / "manifests"
LOCK = REPO / "data" / "data.lock.json"
E1_SOURCES = ["clinc150", "goemotions", "paws", "massive-en", "anli"]


def arbitro_home() -> Path:
    return Path(os.environ.get("ARBITRO_HOME", Path.home() / ".cache" / "arbitro"))


def data_dir() -> Path:
    return arbitro_home() / "data"


def load_manifest(source: str) -> dict:
    return tomllib.loads((MANIFESTS / f"{source}.toml").read_text())


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(source: str) -> dict[str, Path]:
    """Download (or reuse) every file of a manifest and verify it. Returns name -> path."""
    m = load_manifest(source)
    lock = json.loads(LOCK.read_text()) if LOCK.exists() else {}
    out = {}
    for f in m["files"]:
        dst = data_dir() / "raw" / source / f["name"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            print(f"  download {f['url']}", flush=True)
            tmp = dst.with_suffix(dst.suffix + ".part")
            req = urllib.request.Request(f["url"], headers={"User-Agent": "arbitro-data/0.0.1"})
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as w:
                while chunk := r.read(1 << 20):
                    w.write(chunk)
            tmp.rename(dst)
        digest = _sha256(dst)
        pinned = f.get("sha256") or lock.get(source, {}).get(f["name"])
        if pinned and pinned != digest:
            raise RuntimeError(f"{source}/{f['name']}: sha256 {digest} does not match pinned {pinned}; delete {dst} to re-download")
        if not pinned:
            lock.setdefault(source, {})[f["name"]] = digest
            print(f"  TOFU: recorded sha256 {digest[:16]}... for {source}/{f['name']} in {LOCK.relative_to(REPO)}", flush=True)
        out[f["name"]] = dst
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    LOCK.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    return out


def _humanize(label: str) -> str:
    return label.replace("_", " ").strip()


# ------------------------------------------------------------------------------ converters


def convert_clinc150(files: dict[str, Path]) -> tuple[list[dict], dict]:
    d = json.loads(files["data_full.json"].read_text())
    split_map = {"train": "train", "val": "dev", "test": "test", "oos_train": "train", "oos_val": "dev", "oos_test": "test"}
    recs, labels = [], set()
    for key, split in split_map.items():
        for i, (text, intent) in enumerate(d[key]):
            if intent != "oos":
                labels.add(intent)
            recs.append({"id": f"clinc150/{key}/{i}", "source": "clinc150", "split": split, "task": "intent",
                         "state": text, "label": "__none__" if intent == "oos" else intent})
    space = {"labels": sorted(labels), "display": {k: _humanize(k) for k in labels}}
    return recs, space


def convert_goemotions(files: dict[str, Path]) -> tuple[list[dict], dict]:
    names = files["emotions.txt"].read_text().split()
    recs = []
    for fname, split in (("train.tsv", "train"), ("dev.tsv", "dev")):
        with open(files[fname], newline="", encoding="utf-8") as f:
            for row in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
                text, ids, rid = row[0], row[1], row[2]
                recs.append({"id": f"goemotions/{rid}", "source": "goemotions", "split": split, "task": "emotion",
                             "state": text, "labels": [names[int(x)] for x in ids.split(",")]})
    return recs, {"labels": names, "display": {k: k for k in names}}


def convert_paws(files: dict[str, Path]) -> tuple[list[dict], dict]:
    recs = []
    with tarfile.open(files["paws_wiki_labeled_final.tar.gz"]) as tar:
        for split, member in (("train", "final/train.tsv"), ("dev", "final/dev.tsv")):
            f = tar.extractfile(member)
            if f is None:
                raise RuntimeError(f"{member} missing in the PAWS tarball")
            reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE)
            for row in reader:
                recs.append({"id": f"paws/{split}/{row['id']}", "source": "paws", "split": split, "task": "paraphrase",
                             "state": {"sentence1": row["sentence1"], "sentence2": row["sentence2"]},
                             "label": "true" if row["label"] == "1" else "false"})
    return recs, {"labels": ["false", "true"], "display": {"false": "false", "true": "true"}}


def convert_massive_en(files: dict[str, Path]) -> tuple[list[dict], dict]:
    recs, labels = [], set()
    with tarfile.open(files["amazon-massive-dataset-1.1.tar.gz"]) as tar:
        f = tar.extractfile("1.1/data/en-US.jsonl")
        if f is None:
            raise RuntimeError("1.1/data/en-US.jsonl missing in the MASSIVE tarball")
        for line in io.TextIOWrapper(f, encoding="utf-8"):
            r = json.loads(line)
            if r["partition"] not in ("dev", "test"):
                continue  # MASSIVE is evaluation-only (exclusion list, ADR-024)
            labels.add(r["intent"])
            recs.append({"id": f"massive-en/{r['id']}", "source": "massive-en", "split": r["partition"], "task": "intent",
                         "state": r["utt"], "label": r["intent"]})
    return recs, {"labels": sorted(labels), "display": {k: _humanize(k) for k in labels}}


ANLI_LABELS = {"e": "entailment", "n": "neutral", "c": "contradiction"}


def convert_anli(files: dict[str, Path]) -> tuple[list[dict], dict]:
    recs = []
    with zipfile.ZipFile(files["anli_v1.0.zip"]) as z:
        for rnd in ("R1", "R2", "R3"):
            with z.open(f"anli_v1.0/{rnd}/dev.jsonl") as f:
                for line in io.TextIOWrapper(f, encoding="utf-8"):
                    r = json.loads(line)
                    recs.append({"id": f"anli/{rnd}/{r['uid']}", "source": "anli", "split": "dev", "task": "nli",
                                 "state": {"premise": r["context"], "hypothesis": r["hypothesis"]},
                                 "label": ANLI_LABELS[r["label"]]})
    return recs, {"labels": list(ANLI_LABELS.values()), "display": {v: v for v in ANLI_LABELS.values()}}


CONVERTERS = {
    "clinc150": convert_clinc150,
    "goemotions": convert_goemotions,
    "paws": convert_paws,
    "massive-en": convert_massive_en,
    "anli": convert_anli,
}


def prepare(source: str) -> Path:
    files = fetch(source)
    recs, space = CONVERTERS[source](files)
    m = load_manifest(source)
    out = data_dir() / "e1"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{source}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {**space, "allowed_use": m["licence"]["allowed_use"], "licence": m["licence"]["spdx"],
            "licence_status": m["licence"]["status"], "pool": m["source"]["pool"]}
    (out / f"{source}.labels.json").write_text(json.dumps(meta, indent=1))
    counts: dict[str, int] = {}
    for r in recs:
        counts[r["split"]] = counts.get(r["split"], 0) + 1
    print(f"  {source}: {counts} -> {path}", flush=True)
    return path


def load_records(source: str) -> tuple[list[dict], dict]:
    base = data_dir() / "e1"
    path = base / f"{source}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run python -m arbitro_train.data.sources --only {source}")
    recs = [json.loads(line) for line in open(path, encoding="utf-8")]
    meta = json.loads((base / f"{source}.labels.json").read_text())
    return recs, meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", action="append", default=[])
    ap.add_argument("--skip-failed", action="store_true", help="continue when an optional source cannot be downloaded")
    args = ap.parse_args(argv)
    failed = []
    for s in args.only or E1_SOURCES:
        print(f"== {s}", flush=True)
        try:
            prepare(s)
        except Exception as e:
            failed.append(s)
            print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
            if not args.skip_failed:
                return 1
    if failed:
        print(f"failed sources: {failed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
