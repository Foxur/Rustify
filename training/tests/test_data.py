"""CPU tests for the E1 data path (converters, question realisation, layout L0, packing)."""

from __future__ import annotations

import io
import json
import os
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pytest

from arbitro_train.data import sources
from arbitro_train.data.pipeline import NONE_OPTION, NOUL_OPTIONS, L0Builder, Question, item_from, realize
from arbitro_train.packing import StaticShape, fits, pack_items

TOKENIZER_DIRS = [os.environ.get("ARBITRO_TEST_TOKENIZER", ""),
                  "/home/user/_ext/EricYu123456/laya-hexagon-npu/models/tokenizer"]


def _tokenizer():
    for d in TOKENIZER_DIRS:
        if d and Path(d).exists():
            from transformers import AutoTokenizer

            return AutoTokenizer.from_pretrained(d)
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    except Exception:
        pytest.skip("no ModernBERT tokenizer available (set ARBITRO_TEST_TOKENIZER)")


def test_paws_converter(tmp_path):
    rows = "id\tsentence1\tsentence2\tlabel\n1\tA cat sat.\tThe cat sat.\t1\n2\tIt rained.\tIt was sunny.\t0\n"
    p = tmp_path / "paws.tar.gz"
    with tarfile.open(p, "w:gz") as tar:
        for split in ("train", "dev"):
            data = rows.encode()
            info = tarfile.TarInfo(f"final/{split}.tsv")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    recs, space = sources.convert_paws({"paws_wiki_labeled_final.tar.gz": p})
    assert len(recs) == 4 and space["labels"] == ["false", "true"]
    assert recs[0]["label"] == "true" and recs[1]["label"] == "false"
    assert recs[0]["state"] == {"sentence1": "A cat sat.", "sentence2": "The cat sat."}


def test_anli_converter(tmp_path):
    p = tmp_path / "anli.zip"
    with zipfile.ZipFile(p, "w") as z:
        for rnd in ("R1", "R2", "R3"):
            line = {"uid": f"u-{rnd}", "context": "Paris is in France.", "hypothesis": "Paris is in Europe.", "label": "e"}
            z.writestr(f"anli_v1.0/{rnd}/dev.jsonl", json.dumps(line) + "\n")
    recs, _ = sources.convert_anli({"anli_v1.0.zip": p})
    assert len(recs) == 3 and all(r["label"] == "entailment" and r["split"] == "dev" for r in recs)


def test_realize_intent_gold_and_none():
    meta = {"labels": [f"i{k}" for k in range(30)], "display": {f"i{k}": f"intent {k}" for k in range(30)}}
    rng = np.random.default_rng(0)
    for _ in range(200):
        q = realize({"id": "x", "source": "s", "task": "intent", "state": "hi", "label": "i3"}, meta, rng, train=True)[0]
        assert q.options[q.gold] == "intent 3" and 5 <= len(q.options) <= 21
        assert len(set(q.options)) == len(q.options)
        q = realize({"id": "y", "source": "s", "task": "intent", "state": "hi", "label": "__none__"}, meta, rng, train=True)[0]
        assert q.options[q.gold] == NONE_OPTION
    q = realize({"id": "z", "source": "s", "task": "intent", "state": "hi", "label": "i3"}, meta, np.random.default_rng(13), train=False, k_range=(20, 20))[0]
    assert len(q.options) == 20 and NONE_OPTION not in q.options


def test_realize_paraphrase_negation_flips_label():
    meta = {"labels": ["false", "true"], "display": {}}
    rng = np.random.default_rng(1)
    seen = set()
    for _ in range(300):
        q = realize({"id": "p", "source": "paws", "task": "paraphrase", "state": {"sentence1": "a", "sentence2": "b"}, "label": "true"},
                    meta, rng, train=True)[0]
        neg = "differ" in q.instructions or "different" in q.instructions
        assert q.gold == (0 if neg else 1) and q.options == NOUL_OPTIONS
        seen.add(neg)
    assert seen == {True, False}


def test_l0_builder_matches_laya_build_sequence():
    """Layout L0 must be Laya's layout: compare with laya.common.build_sequence when available."""
    laya_common = pytest.importorskip("laya.common")
    tok = _tokenizer()
    b = L0Builder(tok, 512, 192)
    cases = [
        Question("choice", "Which intent does the user express?", ["book hotel", "translate", "none of the listed options applies"], 1,
                 "how do i say hello in german", "c1", "s"),
        Question("noul", "Does the text express joy?", NOUL_OPTIONS, 1, {"ticket": {"id": 7, "body": "Great, thanks! ✨"}}, "c2", "s"),
        Question("choice", "Pick one", [f"option number {i} with a long description text" for i in range(20)], 3, "x " * 600, "c3", "s"),
    ]
    for q in cases:
        ours, markers, spans = b.build(q)
        crit = {o: None for o in q.options} if q.qtype == "choice" else None
        ref_ids, ref_markers = laya_common.build_sequence(tok, q.state, {"t": q.qtype, "ins": q.instructions, "crit": crit}, 512, 192)
        assert ours == ref_ids and markers == ref_markers
        for (s, e), m in zip(spans, markers):
            assert s == m + 1 and e >= s


def test_pack_items_static_shape_and_tail_chunks():
    shape = StaticShape(token_budget=1000, max_segments=40, k_max=4, max_seqlen=128, span_max=300)
    items = [{"ids": list(range(2, 2 + n)), "markers": [1, 3], "spans": [(2, 3), (4, 6)], "qtype": 0,
              "target": [0.0, 1.0], "gold": 1, "src": 0} for n in (50, 90, 30)]
    assert fits(len(items), 170, shape)
    b = pack_items(items, shape)
    cu = b["cu_seqlens"]
    assert len(cu) == shape.max_segments + 1 and cu[-1] == shape.token_budget
    lens = np.diff(cu)
    assert lens.max() <= shape.max_seqlen, "tail chunks must not exceed max_seqlen"
    assert list(lens[:3]) == [50, 90, 30]
    assert b["real_markers"] == 6 and (b["span_owner"] < shape.max_segments * shape.k_max).sum() == 9
    rows = b["marker_rows"][:6]
    assert list(rows) == [1, 3, 51, 53, 141, 143]
    # every chunk restarts positions at 0
    for s, e in zip(cu[:-1], cu[1:]):
        if e > s:
            assert b["position_ids"][s] == 0


def test_item_target_is_one_hot():
    q = Question("choice", "x", ["a", "b", "c"], 2, "s", "r", "src")
    it = item_from(q, ([1, 2, 3, 4, 5], [1, 2, 3], [(2, 2), (3, 3), (4, 4)]), 0)
    assert it["target"] == [0.0, 0.0, 1.0] and it["gold"] == 2
