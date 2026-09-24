"""E1 pipeline: records -> typed questions -> layout-L0 token sequences -> packed micro-batches.

Question realisation (seeded per (seed, epoch, worker), so every epoch sees fresh augmentation):
  intent     choice over 5-20 sampled intents incl. the gold one; a "none of the listed" option in
             30 % of questions and always for out-of-scope utterances (CLINC150 oos -> none)
  emotion    single-label texts: choice over 5-12 sampled emotions (60 %); otherwise a noul question
             "does the text express <e>?" with e gold (true) or a sampled non-gold emotion (false)
  paraphrase noul "same meaning?"; 30 % asked negated ("differ in meaning?") with the label flipped
  nli        (evaluation) choice over entailment / neutral / contradiction with descriptions
Instructions are drawn from a small paraphrase bank. Evaluation questions use a fixed seed.

Layout L0 is Laya's sequence layout (docs/ANALYSIS.md §4.3), built with the backbone's tokenizer:
  [CLS] "<type> question: <instructions>" [SEP] ([MASK] " "+option)... [SEP] state [SEP]
with Laya's budgets: each option <= 48 tokens, the head block <= head_max_len, state truncated
from the right. Every option's text span is recorded for the hybrid read-out.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import torch

from ..packing import StaticShape, fits, pack_items

QTYPE = {"choice": 0, "score": 1, "noul": 2}
NOUL_OPTIONS = ["false: no, the statement does not hold", "true: yes, the statement holds"]
NONE_OPTION = "none of the listed options applies"

INSTRUCTIONS = {
    "intent": ["Which intent does the user express?", "What does the user want to do?",
               "Classify the user's request.", "Which of these best describes what the user is asking for?"],
    "emotion_choice": ["Which emotion does the text express most clearly?", "What is the main emotion of the writer?",
                       "Classify the emotion of this message."],
    "emotion_noul": ["Does the text express {e}?", "Is the writer feeling {e}?", "Does this message show {e}?"],
    "paraphrase": ["Do sentence1 and sentence2 have the same meaning?", "Are the two sentences paraphrases of each other?",
                   "Does sentence2 say the same thing as sentence1?"],
    "paraphrase_neg": ["Do sentence1 and sentence2 differ in meaning?", "Do the two sentences say different things?"],
    "nli": ["What is the relation between the premise and the hypothesis?"],
}
NLI_OPTIONS = ["entailment: the hypothesis follows from the premise",
               "neutral: the premise neither supports nor contradicts the hypothesis",
               "contradiction: the hypothesis contradicts the premise"]
NLI_INDEX = {"entailment": 0, "neutral": 1, "contradiction": 2}


@dataclass
class Question:
    qtype: str
    instructions: str
    options: list[str]
    gold: int
    state: str | dict
    record_id: str
    source: str


def _pick(rng: np.random.Generator, xs: list):
    return xs[int(rng.integers(len(xs)))]


def realize(rec: dict, meta: dict, rng: np.random.Generator, train: bool, k_range=(5, 20)) -> list[Question]:
    task, src = rec["task"], rec["source"]
    display = meta["display"]
    labels = meta["labels"]
    if task == "intent":
        gold_label = rec["label"]
        is_none = gold_label == "__none__"
        k = int(rng.integers(k_range[0], k_range[1] + 1)) if train else k_range[1]
        pool = [lab for lab in labels if lab != gold_label]
        n_distr = k if is_none else k - 1
        distr = list(rng.choice(pool, size=min(n_distr, len(pool)), replace=False))
        opts = distr if is_none else distr + [gold_label]
        opts = [str(o) for o in rng.permutation(opts)]
        rendered = [display[o] for o in opts]
        gold = -1 if is_none else opts.index(gold_label)
        if is_none or (train and rng.random() < 0.3):
            rendered.append(NONE_OPTION)
            if is_none:
                gold = len(rendered) - 1
        ins = _pick(rng, INSTRUCTIONS["intent"]) if train else INSTRUCTIONS["intent"][0]
        return [Question("choice", ins, rendered, gold, rec["state"], rec["id"], src)]
    if task == "emotion":
        golds = rec["labels"]
        if len(golds) == 1 and (not train or rng.random() < 0.6):
            k = int(rng.integers(5, 13)) if train else 8
            pool = [lab for lab in labels if lab not in golds]
            opts = list(rng.choice(pool, size=k - 1, replace=False)) + [golds[0]]
            opts = [str(o) for o in rng.permutation(opts)]
            ins = _pick(rng, INSTRUCTIONS["emotion_choice"]) if train else INSTRUCTIONS["emotion_choice"][0]
            return [Question("choice", ins, opts, opts.index(golds[0]), rec["state"], rec["id"], src)]
        positive = rng.random() < 0.5
        e = str(_pick(rng, golds)) if positive else str(_pick(rng, [lab for lab in labels if lab not in golds]))
        tmpl = _pick(rng, INSTRUCTIONS["emotion_noul"]) if train else INSTRUCTIONS["emotion_noul"][0]
        return [Question("noul", tmpl.format(e=e), NOUL_OPTIONS, int(positive), rec["state"], rec["id"], src)]
    if task == "paraphrase":
        same = rec["label"] == "true"
        negate = train and rng.random() < 0.3
        key = "paraphrase_neg" if negate else "paraphrase"
        ins = _pick(rng, INSTRUCTIONS[key]) if train else INSTRUCTIONS[key][0]
        return [Question("noul", ins, NOUL_OPTIONS, int(same != negate), rec["state"], rec["id"], src)]
    if task == "nli":
        return [Question("choice", INSTRUCTIONS["nli"][0], NLI_OPTIONS, NLI_INDEX[rec["label"]], rec["state"], rec["id"], src)]
    raise ValueError(f"unknown task {task}")


def serialize_state(state) -> str:
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


class L0Builder:
    """Laya-layout sequence builder with token caches (instructions/options by string, states by record id)."""

    def __init__(self, tokenizer, max_len: int = 512, head_max_len: int = 192):
        self.hf = tokenizer
        self.backend = tokenizer.backend_tokenizer
        self.cls, self.sep, self.mask = tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.mask_token_id
        self.pad = tokenizer.pad_token_id
        self.mask_str = tokenizer.mask_token
        self.max_len, self.head_max_len = max_len, head_max_len
        self._str_cache: dict[str, list[int]] = {}
        self._state_cache: dict[str, list[int]] = {}

    def enc(self, s: str) -> list[int]:
        return self.backend.encode(s.replace(self.mask_str, " "), add_special_tokens=False).ids

    def enc_cached(self, s: str) -> list[int]:
        ids = self._str_cache.get(s)
        if ids is None:
            ids = self.enc(s)
            if len(self._str_cache) < 200_000:
                self._str_cache[s] = ids
        return ids

    def state_ids(self, q: Question) -> list[int]:
        ids = self._state_cache.get(q.record_id)
        if ids is None:
            ids = self.enc(serialize_state(q.state))
            self._state_cache[q.record_id] = ids
        return ids

    def build(self, q: Question):
        """Returns (ids, markers, spans) or None if an option marker would be truncated."""
        head = self.enc_cached(f"{q.qtype} question: {q.instructions}")
        opt_ids = [[self.mask] + self.enc_cached(" " + o)[:48] for o in q.options]
        budget = self.head_max_len - sum(len(o) for o in opt_ids)
        if budget < 16:
            per = max(4, (self.head_max_len - 16) // max(1, len(opt_ids)))
            opt_ids = [o[:per] for o in opt_ids]
            budget = self.head_max_len - sum(len(o) for o in opt_ids)
        ids = [self.cls] + head[: max(8, budget)] + [self.sep]
        markers, spans = [], []
        for o in opt_ids:
            markers.append(len(ids))
            spans.append((len(ids) + 1, len(ids) + len(o)))
            ids.extend(o)
        ids.append(self.sep)
        room = max(0, self.max_len - len(ids) - 1)
        ids = ids + self.state_ids(q)[:room] + [self.sep]
        if len(ids) > self.max_len or any(m >= self.max_len for m in markers):
            return None
        return ids, markers, spans


def item_from(q: Question, built, src_index: int) -> dict:
    ids, markers, spans = built
    target = [0.0] * len(markers)
    target[q.gold] = 1.0
    return {"ids": ids, "markers": markers, "spans": spans, "qtype": QTYPE[q.qtype], "target": target,
            "gold": q.gold, "src": src_index, "weight": 1.0}


def load_tokenizer(name_or_dir: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name_or_dir)
    for attr in ("cls_token_id", "sep_token_id", "mask_token_id", "pad_token_id"):
        if getattr(tok, attr) is None:
            raise ValueError(f"tokenizer {name_or_dir} has no {attr}")
    return tok


class TrainStream(torch.utils.data.IterableDataset):
    """Infinite stream of packed micro-batches. Sources are mixed by temperature sampling
    (p ∝ n^tau per epoch); each DataLoader worker handles a disjoint slice of every epoch."""

    def __init__(self, sources: dict[str, tuple[list[dict], dict]], tokenizer_name: str, shape: StaticShape,
                 max_len: int, head_max_len: int, seed: int, tau: float = 0.4, lookahead: int = 256):
        self.sources = sources
        self.names = sorted(sources)
        self.tokenizer_name = tokenizer_name
        self.shape, self.max_len, self.head_max_len = shape, max_len, head_max_len
        self.seed, self.tau, self.lookahead = seed, tau, lookahead
        sizes = np.array([len(sources[n][0]) for n in self.names], dtype=float)
        p = sizes ** tau
        self.per_epoch = np.round(p / p.sum() * sizes.sum()).astype(int)

    def epoch_order(self, epoch: int) -> list[tuple[int, int]]:
        rng = np.random.default_rng([self.seed, epoch, 7])
        order = []
        for si, n in enumerate(self.names):
            size = len(self.sources[n][0])
            want = int(self.per_epoch[si])
            idx = np.concatenate([rng.permutation(size) for _ in range(-(-want // size))])[:want]
            order += [(si, int(i)) for i in idx]
        perm = rng.permutation(len(order))
        return [order[i] for i in perm]

    def __iter__(self):
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        info = torch.utils.data.get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        builder = L0Builder(load_tokenizer(self.tokenizer_name), self.max_len, self.head_max_len)
        epoch = 0
        pending: list[dict] = []
        used = 0
        while True:
            rng = np.random.default_rng([self.seed, epoch, wid, 11])
            order = self.epoch_order(epoch)[wid::nw]
            for si, ri in order:
                recs, meta = self.sources[self.names[si]]
                for q in realize(recs[ri], meta, rng, train=True):
                    built = builder.build(q)
                    if built is None:
                        continue
                    it = item_from(q, built, si)
                    n = len(it["ids"])
                    if not fits(len(pending) + 1, used + n, self.shape):
                        yield {**pack_items(pending, self.shape), "epoch": epoch}
                        pending, used = [], 0
                    pending.append(it)
                    used += n
            epoch += 1


def build_eval_batches(records: list[dict], meta: dict, src_index: int, builder: L0Builder, shape: StaticShape,
                       seed: int = 13, limit: int | None = None) -> tuple[list[dict], int]:
    """Deterministic evaluation micro-batches for one source/split. Returns (batches, n_questions)."""
    rng = np.random.default_rng(seed)
    k_range = (20, 20) if records and records[0]["task"] == "intent" else (5, 20)
    batches, pending, used, n_q = [], [], 0, 0
    for rec in records[:limit] if limit else records:
        for q in realize(rec, meta, rng, train=False, k_range=k_range):
            built = builder.build(q)
            if built is None:
                continue
            it = item_from(q, built, src_index)
            if not fits(len(pending) + 1, used + len(it["ids"]), shape):
                batches.append(pack_items(pending, shape))
                pending, used = [], 0
            pending.append(it)
            used += len(it["ids"])
            n_q += 1
    if pending:
        batches.append(pack_items(pending, shape))
    return batches, n_q
