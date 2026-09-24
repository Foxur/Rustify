# Architecture decisions: Foxur/Rustify (working product name "Arbitro")

Date: 2026-09-23. Author: lead architect. Audience: the solo maintainer (one RTX 4090 24 GB, about 12 h/week), and every downstream document.

**What this file is.** It is the single source of truth that turns the three design proposals and the two design-review panels into decisions ADR-001 to ADR-033. Where this file disagrees with a proposal, this file wins. Where a research report disagrees with `critic.md`, `critic.md` wins (for example: every shipped Laya checkpoint runs bf16 autocast on sm_89, weights are fp16 on disk, and Jev choice confidence is (n·peak−1)/(n−1)).

**Where it lives.** The repository keeps this file as `docs/DECISIONS.md`; the companion documents are [ARCHITECTURE.md](ARCHITECTURE.md), [ROADMAP.md](ROADMAP.md) and [TRAINING.md](TRAINING.md), plus the reference analysis of Jev and Laya, [ANALYSIS.md](ANALYSIS.md), which quotes §5 unchanged. The proposals, the review panels and the research reports it cites are design-phase inputs that are not in the repository (Q15). A documentation consistency review on 2026-09-23 corrected errata in place and added Q15–Q18; Appendix C lists every amendment.

**Binding rules for downstream documents**
- Use the names in §3, the terms in §4 and the numbers in §5 exactly as written. When a value changes, change §5 first, then the prose.
- Cite evidence with the keys in Appendix B, for example [CR C1] or [RIS §5.3].
- Label every number with one of these:
  - **VERIFIED**: read in source or recomputed from raw data, by the research reports or during the design phase.
  - **REPORTED**: a third-party claim.
  - **ESTIMATED**: modelled, not measured. The M0 or M4 measurements replace it.
  - **MEASURED**: measured by this project. Nothing has that status yet.
  - **GATE** / **GOAL**: a target. A GATE blocks a release; a GOAL does not.
- Anything nobody has checked is **UNVERIFIED**.

**Contents**
1. Executive summary
2. Decisions needing the maintainer's input
3. Canonical names
4. Canonical glossary
5. Canonical numbers
6. Resolved disagreements (index)
7. ADR-001 … ADR-033
8. Appendix A: review-panel findings and how each was fixed
9. Appendix B: evidence keys
10. Appendix C: amendments from the documentation review

---

## 1. Executive summary

**What we build.** A self-hosted, Apache-2.0 typed-decision engine written in Rust. The name is **Arbitro** (Q1, decided 2026-09-24); the GitHub repo stays `Foxur/Rustify` for now (Q1). The engine:
- speaks the Jev wire format (`POST /v1/systemone`, `GET /v1/models`), so the unmodified TypeSafe SDKs work with only `TYPESAFE_BASE_URL` changed;
- runs user-downloaded Laya checkpoints with verified numerical parity;
- later serves our own licence-clean models (the **dm2** family) trained on the RTX 4090.

These are two products with two results tables, and the tables are never merged (ADR-001).

**Sequencing** (ADR-006, ADR-032). Estimates assume about 12 h/week; the hours already include the review panels' 1.5× corrections.

| Release | Cumulative dev-h | Week | Contents |
|---|---|---|---|
| v0.1 "drop-in" | 158 | ~14 | CPU fp32 plus candle-CUDA bf16, our own ModernBERT code, the Jev-compatible server (`strict`/`lenient`/`laya` modes), Laya parity L0–L3, the SDK conformance suite, Docker |
| v0.2 "engine" | 364 | ~31 | Custom cudarc + cuBLASLt + vendored FlashAttention-2 engine with the laya-v1 kernel subset (170 dev-h budget), zero-delay batching, CUDA graphs, batch-invariant by default |
| v0.3 | 505 | ~43 | First own model, `arbitro-en-large`, after pre-registered ablations |
| 1.0 | 535 | ~45 | API freeze |

Multilingual weights (pending counsel on the tokenizer) and FP8 come after 1.0.

**Engine numerics** (ADR-009, ADR-014).
- GPU default for every checkpoint: bf16 GEMM inputs with fp32 accumulate. The residual stream, LayerNorm, softmax, head and scorer stay fp32.
- fp16 is opt-in only after a max-abs sweep. FP8 comes later via CUTLASS sm89, because cuBLASLt on Ada supports only scalar scales.
- GPU parity against the PyTorch CUDA bf16 reference: mean |Δp| ≤ 2e-3 and max |Δp| ≤ 2e-2. CPU fp32 parity: ≤ 1e-4.

**Speed** (ESTIMATED; re-based on M0 measurements).
- Gates: ≤ 6 ms p50 per 250-token question in-process, and ≥ 350 q/s saturated.
- Goals: 3–4 ms and ≥ 450 q/s.
- That is about 2.5× the PyTorch throughput on the same card, not "10×".

**Serving defaults** (ADR-015 to ADR-018).
- `lenient` mode.
- Full-precision probabilities with a 1e-6 floor, so there are no hard zeros.
- Rescaled-peak confidence for choice questions, peak confidence for score questions.
- An 8 s deadline, below the SDK's 10 s per-attempt timeout.
- 503 for overload (529 in `strict`); 429 only for per-key rate limits.
- `usage` counts the tokens actually processed.

**Own model, dm2** (ADR-019 to ADR-022).
- Read-out from the pretrained `[MASK]` marker, optionally combined with span pooling.
- Shared-state layout L2 if it stays within 1 pp of the per-question layout L0. Otherwise layout L3 (late fusion), and layout L0 as the last fallback. The isolated-option tree is only an ablation arm.
- Per-option spans with option-group chunking up to 255 options, and an explicit "none" option.
- Two output channels: a calibrated distribution, plus an out-of-fold `p_correct`.
- Backbone chosen by A/B, with an early signal by about week 11 (E1). DeBERTa-v3-large is adopted only as a teacher: if it wins by more than 3 pp, we distil it into a ModernBERT-family student.
- Primary release gate: +10 pp macro accuracy over `laya-en` on held-out sources, with the CI lower bound > +5 pp.

**Training and data** (ADR-023 to ADR-026).
- A PyTorch 2.14 trainer. Rust owns data, packing, calibration and evaluation via PyO3, so training and serving use the same sequence builder.
- Soft labels only from Apache/MIT open-weight teachers. Never Jev outputs; never Laya weights or outputs.
- Four data pools: train, held-out-source, Jev-comparable and Laya reproduction. Training is disjoint from the other three at source level, checked by MinHash. The never-trained sources include Banking77, AG News, DAIR emotion, SMS spam, MMLU, WANLI, all of MASSIVE, and typed-decisions.
- 126–193 GPU-h to v0.3, or 158–241 with 25 % slack.

**Legal** (ADR-030, ADR-031).
- Apache-2.0 with DCO. Vendored FA2/CUTLASS carry file-level provenance.
- No Jev marks in identifiers: the modes are `strict`/`lenient`/`laya` and the method is `decide`.
- No TypeSafe account, ever.
- Laya weights are never redistributed and never used to initialise our models.

**Top risks** (ADR-033).
1. Solo bandwidth. Explicit cut rules handle it.
2. ModernBERT-large cold start, rated M-H/H. E1 gives the early signal.
3. dm2 not clearly beating Laya.
4. The custom engine overrunning. candle-CUDA stays shippable meanwhile.

**Needed from the maintainer now:** Q3 typed-decisions reproduction, Q4 CC-BY-SA data, Q5 platforms, Q6 hardware and runner, Q7 time budget. Before M2: Q16 (CUTLASS fetch in the `candle-cuda` build), Q17 (fp32 GPU path for T5/T9), Q18 (admission vs backpressure). Before the first public announcement: Q15 (publishing the research reports). By v0.1: Q14, the order of engine vs model (§2).

---

## 2. Decisions needing the maintainer's input

Each question has a default. The project proceeds on the default until the maintainer answers.

| # | Question | Recommendation / default if unanswered | Needed by | ADR |
|---|---|---|---|---|
| Q1 | **ANSWERED 2026-09-24: Arbitro.** **Name.** Options: (a) "Arbitro": free on crates.io, PyPI and npm, re-checked 2026-09-23; trademark search UNVERIFIED. (b) "Rustify": the crate `rustify` is an unrelated HTTP client with 12.4M downloads, so our crates would be `rustify-*` and the facade would be `rustify-decide`. (c) Another name. | Arbitro. No `cargo publish` until answered. | M0 exit (week 2); at the latest before the first publish (M2) | ADR-002 |
| Q2 | **ANSWERED 2026-09-24: Apache-2.0 only.** **Licence.** Apache-2.0 only, or MIT OR Apache-2.0? | Apache-2.0 only, because we vendor or derive Apache-only code | M0 | ADR-030 |
| Q3 | **typed-decisions.** May `LocalLLaMA/typed-decisions` be used privately, never published, to reproduce Laya's fine-tune ("P1") from Laya weights you downloaded? Its licence and teacher are unknown [CR G14]. | **No.** Validate the trainer with a licence-clean reproduction instead (ADR-023). | M3b (week ~31) | ADR-023, ADR-024 |
| Q4 | **Share-alike data.** May CC-BY-SA or CDLA-Sharing sources train Apache-2.0 weights? Examples: SNLI, BoolQ, ARC, FEVER, SGD, SIB-200, Belebele, Civil Comments text. | **Exclude** them from training. They stay usable for evaluation. | M3a (week ~15) | ADR-024 |
| Q5 | **Platforms.** Which tiers? | Tier-1: Linux x86_64 CPU and NVIDIA CUDA. Tier-2: macOS arm64 (CPU/Metal) and Linux aarch64 CPU. Tier-3: Windows x64 CPU (build-only). | M2 | ADR-006, ADR-029 |
| Q6 | **Hardware and operations.** Is the 4090 headless (a display costs 0.3–1 GB of VRAM)? Which CPU (AVX-512?), how much RAM and free disk (≥ 1 TB recommended)? Is 24/7 operation at 350–380 W acceptable? May the 4090 act as a self-hosted CI runner (`main` and scheduled jobs only)? | Assume a desktop that is headless at night, and a runner that only runs nightly | M0 | ADR-026, ADR-029 |
| Q7 | **Time budget.** Is ~12 h/week realistic? | At 8 h/week, 1.0 moves from ~week 45 to ~week 67 (~15.5 months) | M0 | ADR-032 |
| Q8 | **Multilingual.** Which languages matter? What is your risk tolerance on mmBERT's Gemma-2-derived tokenizer? Is a lawyer available? | Release English-only weights until counsel clears the tokenizer | M7 | ADR-020, ADR-030 |
| Q9 | **Distribution.** Which Hugging Face org for our weights? Is `ghcr.io/foxur` right? Do you want a Homebrew tap? | `ghcr.io/foxur`; HF org to be decided; no tap before 1.0 | M2 (images), M6 (weights) | ADR-030 |
| Q10 | **Outreach.** Contact Convai (Laya) and the authors of the `laya` / `laya-rs` crates about the typed-decisions licence, fixtures or collaboration? | Yes, informational only, after v0.1 | v0.1 | ADR-033 |
| Q11 | **Determinism.** Keep batch-invariant determinism on by default even if it costs up to 10 % throughput? | Yes | M4 | ADR-011 |
| Q12 | **Post-1.0 accurate tier.** Is a 4–8B decoder tier of interest after 1.0? Is there any cloud budget for a larger teacher? | No; it is a non-goal before 1.0 | before 1.0 | ADR-001, ADR-025 |
| Q13 | **Commercial use.** Is it planned, by you or by expected users? | Assume yes, so apply the conservative licence defaults | M3a | ADR-024, ADR-030 |
| Q14 | **Order after v0.1.** Engine first (v0.2 engine at ≈ week 31, first own model at ≈ week 43), or model first (own model at ≈ week 27, engine at ≈ week 43)? | Engine first | v0.1 (week ~14) | ADR-032 |
| Q15 | **Research reports.** The evidence keys ([LIS], [JAS], [BW], …, Appendix B) point to design-phase reports that are not in the repository. They contain analyses of third-party Jev logs and REPORTED terms-of-service quotes. Publish them? | Publish them under `docs/research/` after a review that removes verbatim third-party content beyond short quotations. Until then the keys stay, and every document says that the reports are not yet public. | Before the first public announcement | ADR-027, ADR-031 |
| Q16 | **CUTLASS fetch in a dependency.** `candle-flash-attn` 0.11 (used by `candle-cuda`) fetches CUTLASS at build time (its `build.rs`, via cudaforge, pins commit `7d49e6c7`; VERIFIED). Does ADR-030's "no build-time fetch" also bind dependencies? | Accept the dependency for the v0.1 `candle-cuda` backend, but release and image builds pre-seed a sha256-checked CUTLASS checkout at that commit (cudaforge reuses an existing checkout under `$CUDAFORGE_HOME`, VERIFIED in its `src/dependency.rs`), so they never touch the network. ADR-030's rule binds our own crates. | Before M2 | ADR-030 |
| Q17 | **fp32 on the GPU for T5 and T9.** T5 (Rust CUDA fp32 vs Rust CPU fp32), L2's "T5 on CUDA" and T9's fp32 paged-vs-dense clause need fp32 attention on the GPU, but the vendored FA2 kernels (K3, K3b) and candle-flash-attn exist only in fp16/bf16 (VERIFIED). Add a debug fp32 path, or restate the gates? | Add a **debug-only fp32 path** that is never a serving mode: `candle-cuda` runs fp32 through the unfused masked attention that `arbitro-candle` already needs for `metal`; `cuda` adds fp32 variants of its element-wise kernels and K13 (a naive fp32 varlen windowed reference attention, with a block-table mode from M6), inside M4's 170 dev-h (UNVERIFIED that it fits; C-3 still applies). T5 and T9's fp32 clauses run on this path; K3/K3b are covered by their float64 unit tests and by T4. | Before M2 (v0.1 T5) | ADR-008, ADR-014 |
| Q18 | **Admission vs backpressure.** An `auto` `max_processed_tokens` (e.g. 262,144 at 128k tok/s) can exceed S10 = 131,072 queued tokens, so a request that passes the limit could be refused with 503/529 even by an idle server. | Cap `auto` at `max_queued_tokens` (S10). A request is refused with 503/529 when queued tokens plus its processed tokens exceed S10, so an idle server admits every request that passes the limits. | Before M2 | ADR-010, ADR-017 |

---

## 3. Canonical names (downstream documents MUST use these)

### 3.1 Project, crates, binaries, packages

| Thing | Canonical name | Notes |
|---|---|---|
| Product (display) | **Arbitro** | Decided 2026-09-24 (Q1). |
| Lower-case token | `arbitro` | The root of every identifier below. |
| Repository | `github.com/Foxur/Rustify` | Unchanged for now; a later rename to `arbitro` is optional (GitHub redirects). "Rustify" is only the repo's codename: never a crate, binary or model name. |
| Published crates (8) | `arbitro-proto`, `arbitro-core`, `arbitro-compat`, `arbitro-candle`, `arbitro-cuda`, `arbitro-server`, `arbitro-eval`, `arbitro` | ADR-003. Versions are lockstep. |
| Internal crates (`publish = false`) | `arbitro-data`, `arbitro-py`, `xtask` | `arbitro-py` builds the PyPI wheel `arbitro`. |
| Reserved crate | `arbitro-ort` | Created only if ADR-007's CPU spike fails. |
| Binary | `arbitro` | Built from the facade crate (feature `cli`, on by default). |
| CLI subcommands | `serve`, `decide`, `pull`, `models`, `inspect`, `calibrate`, `eval`, `bench`, `parity`, `sweep-overflow`, `export-check`, `doctor`, `config` | `export-onnx` is reserved for ADR-007's ORT path. No other names: not `run`, `fetch` or `predict`. |
| xtask commands | `cargo xtask goldens`, `parity`, `numbers`, `fetch-test-assets`, `release-check` | |
| Python package | PyPI `arbitro`, import `arbitro` | Exposes `arbitro.Engine`, `arbitro.compat.laya.Agent` (migration shim), `arbitro.data`, `arbitro.calib`, `arbitro.eval`. |
| Docker images | `ghcr.io/foxur/arbitro:<version>-cpu`, `ghcr.io/foxur/arbitro:<version>-cuda` | Never contain weights. |
| Trainer project | `training/` (uv project `arbitro_train`) | |
| Golden generator | `tools/goldens/` (uv project) | Pinned reference environment (ADR-012). |

**Rename rule (if Q1 picks another name).** Replace the token mechanically:
- `arbitro` → `<name>`, `Arbitro` → `<Name>`, `ARBITRO` → `<NAME>`, and `x_arbitro` → `x_<name>`.

If the name stays "Rustify", then:
- the crates are `rustify-<x>` (all free);
- the facade crate is `rustify-decide`, because `rustify` is taken;
- the binary and the PyPI package are `rustify`.

Nothing else changes.

### 3.2 Rust API identifiers (ADR-004)

| Kind | Names |
|---|---|
| Facade | `arbitro::Engine`, `Engine::builder()`, `Engine::decide` (async), `Engine::decide_blocking`, `Engine::models` |
| Wire types (`arbitro-proto`) | `DecideRequest`, `DecideResponse`, `Question` (`Choice`/`Score`/`Noul`), `ChoiceCriteria`, `Answer`, `Usage`, `RequestExt`, `ResponseExt`, `AnswerExt`, `ApiError` |
| Core traits and structs (`arbitro-core`) | `Backend`, `Runner`, `Frontend`, `PackedBatch`, `KvSegment`, `RawOutputs`, `WorkPlan`, `Precision` (`F32`, `Bf16`, `Fp16Checked`, `Fp8W8A8`), `ModelArtifacts`, `Registry` |
| Never used | `system_one` (a Jev product mark) and `predict`. The method is always `decide`. |

### 3.3 Configuration (`arbitro.toml`)

Precedence, lowest to highest: `arbitro.toml` → environment `ARBITRO__<SECTION>__<KEY>` → CLI flags.

Other environment names:
- API keys: `ARBITRO_API_KEYS` (comma-separated).
- Cache root: `ARBITRO_HOME`, default `~/.cache/arbitro`.
- User registry: `~/.config/arbitro/models.toml`.

The canonical defaults are:

```toml
[server]
bind = "127.0.0.1:8080"         # the Docker images set 0.0.0.0:8080 (ADR-017)
mode = "lenient"                 # strict | lenient | laya
request_timeout_ms = 8000
max_body_bytes = 8388608
max_concurrent_requests = 512
path_aliases = false             # optional /api/alpha/decisions and /typesafe/v1/* aliases

[auth]
api_keys_env = "ARBITRO_API_KEYS"
api_keys_file = ""               # file of SHA-256 key hashes
missing_key_status = 401         # 403 in strict mode

[models]
default = "laya-en"
preload = ["laya-en"]
max_resident = 3
routing = "explicit"             # explicit | laya-heuristic (v0.2) | lid (dm2 multilingual)
aliases = { "jev-latest" = "@default", "jev-preview" = "@default" }  # accepted input values (wire compatibility)
list_aliases = false             # aliases are not listed in GET /v1/models by default

[engine]
backend = "auto"                 # auto | cpu | cuda | candle-cuda | metal
device = 0
precision = "auto"               # auto (bf16 on GPU, fp32 on CPU) | fp32 | bf16 | fp16 | fp8
determinism = "batch_invariant"  # batch_invariant | fast
cuda_graphs = true
threads = 0                      # 0 = physical cores

[scheduler]
max_batch_tokens = 16384         # 32768 for base-size models
max_wait_us = 0                  # zero-delay batching
max_queued_tokens = 131072       # 8 x max_batch_tokens
priorities = true                # interactive | bulk

[limits]
max_options = 255
max_score_levels = 10
max_request_tokens = 65536
max_state_plus_question_tokens = 32768
max_processed_tokens = "auto"    # 0.5 x deadline x measured tok/s, power-of-two floor, min 4096, capped at max_queued_tokens (Q18)

[output]
rounding = "@mode"               # full (lenient) | round2 (strict) | round4 (laya)
prob_floor = 1e-6
score_confidence = "peak"        # peak | rescaled_peak
extensions = true

[observability]
log_format = "json"
metrics = true
log_bodies = false
otlp_endpoint = ""
```

### 3.4 Wire-level identifiers (ADR-015 to ADR-018)

| Kind | Canonical |
|---|---|
| Routes | `POST /v1/systemone`, `GET /v1/models`, `GET /health`, `GET /ready`, `GET /metrics`, `GET /openapi.json`. Optional, off by default: `POST /api/alpha/decisions` and `/typesafe/v1/*`. |
| Extension key | `x_arbitro`. It is a top-level request field and a top-level response field. Per-answer extras live at `x_arbitro.answers.<qid>`, never inside the answer objects. |
| Headers | `x-typesafe-request-id: req_<32 hex>` is wire-required and always sent. Also: `x-request-id` (echoed), `x-arbitro-model`, `x-arbitro-truncated-questions`, `server-timing`, `retry-after`, `retry-after-ms`. |
| Server modes | `strict`, `lenient` (default), `laya` |
| Rounding | `full`, `round2`, `round4` |
| Score confidence | `peak` (default), `rescaled_peak` |
| Backends | `cpu`, `cuda`, `candle-cuda`, `metal` (`ort` reserved) |
| Precision values | `auto`, `fp32`, `bf16`, `fp16`, `fp8` |
| Determinism | `batch_invariant` (default), `fast` |
| Routing | `explicit`, `laya-heuristic`, `lid` |
| Metrics prefix | `arbitro_`, e.g. `arbitro_requests_total{model,status}` |

### 3.5 Model identifiers and families (ADR-005)

| Registry id | Family | Source (pinned) | Concrete id returned in `model` | Budgets `max_len` / `head_max_len` |
|---|---|---|---|---|
| `laya-en` | `laya-v1` | HF `convaiinnovations/laya` (root) @ `c5d78730f3493e4fe16d61507ef4b78eef7318cf` | `laya-en-c5d78730` | 512 / 192 |
| `laya-multilingual` | `laya-v1` | HF `convaiinnovations/laya`, subfolder `multilingual` @ `1c5edc17a7acd8701df6fc341c0d179f1c62c982` | `laya-multilingual-1c5edc17` | 1024 / 256 |
| `laya-typed-decisions` | `laya-v1` | HF `convaiinnovations/laya-typed-decisions` @ `f9ab0b228f0fc0f14d873dbc99038f135c2da1b2` | `laya-typed-decisions-f9ab0b22` | 1024 / 256 |
| `test-tiny-en`, `test-tiny-multi` | `laya-v1` | Random weights, generated deterministically by `tools/goldens` (never Laya weights) | same as the id | small, test-only; listed only with the `test-models` feature |
| `arbitro-en-large` | `dm2` | Our Hub org (Q9) | `arbitro-en-large-<model semver>`, e.g. `arbitro-en-large-1.0.0` | 8192 context |
| `arbitro-en-base` | `dm2` | same | `arbitro-en-base-<semver>` | 8192 context |
| `arbitro-multi-base` | `dm2` | same, only after counsel clears the tokenizer (Q8) | `arbitro-multi-base-<semver>` | 8192 context |

- **Response `model`.** In `strict` and `lenient` mode it is the concrete id above, never an alias. `laya` mode returns laya-serve's `"laya-rl-agent"` for byte parity (ADR-015). The `x-arbitro-model` header carries the concrete id in every mode.
- **Aliases in `laya` mode** (the laya-serve names): `english`, `en` and `laya` → `laya-en`; `multilingual` → `laya-multilingual`; `typed-decisions` → `laya-typed-decisions`.
- **Model versions** use their own semver, starting at `1.0.0` for the first released weights. They are independent of software versions: the first own model ships in software v0.3.0.
- **"v2".** Downstream documents MUST NOT say "v2" on its own. Say "own-model track", "dm2" (the family) or the model id.

### 3.6 Repository layout

```
Cargo.toml  rust-toolchain.toml  deny.toml  LICENSE  NOTICE  README.md  CONTRIBUTING.md  SECURITY.md
crates/{arbitro-proto,arbitro-core,arbitro-compat,arbitro-candle,arbitro-cuda,arbitro-server,arbitro-eval,arbitro,arbitro-data,arbitro-py}
xtask/
third_party/{flash-attention,cutlass,vllm-scaled-mm}/   + third_party/README.md (file-level provenance)
tools/goldens/           training/            tests/{conformance,goldens}/   examples/
evals/{registry.toml,registrations/}          data/manifests/               reports/{*.json,claims.toml,spikes/}
docs/ (mdBook)  docs/{ANALYSIS,DECISIONS,ARCHITECTURE,ROADMAP,TRAINING}.md   docs/adr/ADR-NNN-<slug>.md
                docs/clean-room.md   docs/ablations.md   docs/perf-baseline.md   docs/ci.md
```

---

## 4. Canonical glossary

| Term | Meaning |
|---|---|
| **Typed decision** | One question answered with a calibrated probability distribution, never with generated text. |
| **choice / score / noul** | The three question types. choice: one of 1–255 named options. score: an ordinal scale of 1–10 levels; `score` = Σ i·pᵢ. noul: yes/no; the answer is P(yes). |
| **state** | The context the questions are about: a string, JSON object or JSON array. |
| **criteria / instructions** | criteria are the options (choice), the levels (score) or the true/false descriptions (noul). instructions is the question text. |
| **Jev** | TypeSafe AI's hosted "System One" model. It is only ever mentioned nominatively ("Jev-compatible wire format"). |
| **Laya** | Convai Innovations' open-weights Jev-compatible model, reference implementation laya 0.3.7. |
| **compat runtime** | Product track A: running user-downloaded Laya checkpoints with verified parity. |
| **own-model track** | Product track B: our `dm2` models. |
| **laya-v1 (family)** | Laya's architecture and sequence layout: one sequence per question, 2-layer head, `[MASK]` marker per option. |
| **dm2 (family)** | Our own model architecture (ADR-019). |
| **laya-plus** | Opt-in fixes applied to Laya checkpoints, reported separately and never called parity: held-out recalibration via `arbitro calibrate`, `x_arbitro.permutations`, and the noul empty-state prior correction. |
| **Parity levels L0–L5** | The parity harness (ADR-014). Do not confuse them with layouts L0/L2/L3, which are always written "layout L0" and so on. |
| **Layout L0 / L2 / L3-k / T** | dm2 sequence layouts. L0: per-question sequence, Laya-style. L2: prefix-isolated shared state. L3-k: late fusion over the top k layers. T: the isolated-option tree, an ablation arm only. |
| **Option-group chunking** | Splitting a choice question into groups of ≤ 64 options. Each group is a suffix over the same state, and all logits go through one joint softmax. |
| **none option** | A virtual last option ("none of the options applies"). Its mass is `p_none`. |
| **Dual channel** | Channel 1 is the calibrated option distribution. Channel 2 is `p_correct`, from an out-of-fold correctness head. |
| **Rescaled-peak confidence** | (n·p_max − 1)/(n − 1), clipped to [0, 1]; 1.0 when n = 1. This is Jev's documented choice confidence [CR C4]. |
| **Peak confidence** | p_max. The default score confidence [CR G7]. |
| **Entropy confidence** | 1 − H/ln k. Laya's statistic; used only in `laya` mode. |
| **Processed tokens** | Tokens that actually go through the encoder. laya-v1 re-encodes the state once per question. `usage.input_tokens` reports this number. |
| **Zero-delay batching** | Launch as soon as the GPU is idle; pack the next batch while it is busy. |
| **Batch invariance** | Bitwise-identical outputs whether a request runs alone or co-batched (ADR-011). |
| **Pools T / O / J / L** | T: training sources. O: held-out-source (OOD-S) evaluation. J: Jev-comparable suites, test-only. L: Laya reproduction. Pools T and O ∪ J ∪ L are disjoint at source level; pools O and J are disjoint (ADR-024). |
| **OOD-S suite** | The held-out-source suite of pool O. Its test macro accuracy is the primary endpoint. |
| **Contamination tags** | `in-train`, `held-out-source` (the family is seen in training), `held-out-family`. Every result row carries one. |
| **E1** | The early model-signal spike (weeks 5–11): backbone and layout signal on a gold-only mini-mixture. |
| **X1–X7** | The pre-registered ablations in M5 (ADR-019, ADR-020). |
| **G-Q1…G-Q6** | The dm2 release gates (ADR-022). |
| **P#, MEM#, S#, T#, Q-ref#, G-Q#, C#, R-#** | IDs in the canonical numbers table (§5). |
| **M0…M9, M3a/M3b, M4c** | Milestones (ADR-032). |
| **K1…K13, AM-#, O-#** | Kernels of the custom engine (ADR-008); amendments from the documentation review (Appendix C); open items of ARCHITECTURE.md §16. |
| **dev-h / GPU-h** | Maintainer hours / RTX 4090 hours. |
| **GPU night** | The unattended overnight window used for training and labelling (ADR-026). |

---

## 5. Canonical numbers (downstream documents MUST quote these identically)

Every value carries its kind. All values marked ESTIMATED are re-based after the M0 and M4 measurements. A re-base changes this table first and is recorded in `reports/`.

### 5.1 Performance (RTX 4090, `laya-en` ModernBERT-large, warm, unless noted)

| ID | Quantity | Value | Kind | Evidence |
|---|---|---|---|---|
| P1 | 1 question × 250 tokens, in-process p50, `cuda` engine, bf16 | ≤ 6 ms (gate, v0.2); 3–4 ms (goal) | GATE / GOAL, ESTIMATED | [RIS §5.3], [CR C6] |
| P2 | Same over HTTP loopback | p50 ≤ 8 ms and p99 ≤ 15 ms (gate); p50 ≤ 5 ms (goal) | GATE / GOAL, ESTIMATED | [RIS §5.3], [JAS §4] |
| P3 | Saturated throughput, 250-token questions, bf16 | ≥ 350 q/s (gate); ≥ 450 q/s (goal); planning figure ~500 q/s | GATE / GOAL, ESTIMATED | [CR C6], [RIS §5.3] |
| P4 | 1 request × 50 questions × 250 tokens (12.5k processed tokens, laya-v1 layout) | ≤ 150 ms (gate); ≤ 100 ms (goal) | GATE / GOAL, ESTIMATED | ~128k tok/s [RIS §5.3] |
| P5 | Custom engine throughput vs the PyTorch reference on the same 4090 | ≈ 2.5× tokens/s (goal); no "10×" claim anywhere | GOAL, ESTIMATED | 128k vs 52.3k tok/s [RIS §5.3] |
| P5a | PyTorch ModernBERT-large on a 4090 | 52.3k tok/s (512 fixed length) | REPORTED (ModernBERT paper) | [RIS §5.3] |
| P6 | v0.1 `candle-cuda`, 1 × 250 tokens, in-process p50 | ≤ 12 ms **and** ≤ the PyTorch-reference p50 measured in M0 (gate); saturated q/s ≥ PyTorch reference (gate); estimate 6–10 ms | GATE, ESTIMATED | [RIS §5.3] ("~8–10 ms" at L=512) |
| P7 | `cpu` backend, 8 physical cores, 1 × 250 tokens, fp32 | ≤ 1.5× PyTorch CPU fp32 latency on the same machine (gate); ≤ 1.0× (goal) | GATE / GOAL | candle's default gemm is 3.3–3.9× slower than PyTorch (VERIFIED on a 4-core Xeon) [RIS §2.6] |
| P8 | `laya-multilingual` (mmBERT-base) cost per token vs EN | ≈ 2.8× cheaper | ESTIMATED | [RIS §5.2] |
| P9 | dm2 layout L2: 500-token state + 10 questions × 60 tokens (instructions + 4 options) | p50 ≤ 1.5× the p50 of the same state with 1 question (gate); ≤ 1.2× (goal) | GATE / GOAL, ESTIMATED | ADR-019 |
| P9a | dm2 layout L2: 600-token state + 5 × 120-token questions (1,200 processed tokens) | p50 ≤ 12 ms (goal) | GOAL, ESTIMATED | 0.78 GFLOP/token [RIS §5.2] |
| P10 | Cost of `batch_invariant` vs `fast` | ≤ 10 % of saturated throughput (it stays the default either way, Q11) | GATE (triggers a post-1.0 fixed-K GEMM item), UNVERIFIED | [RIS §6.7] |
| P11 | FP8 W8A8 (1.x only) | ≥ 650 q/s (goal; estimate ~800). Gates vs bf16: argmax agreement ≥ 99.5 %, max \|Δp\| ≤ 0.02, ΔNLL ≤ +0.01 nats and ΔECE ≤ +0.005 after a per-precision refit, flip rate non-inferior | GOAL / GATE, ESTIMATED | [RIS §5.3, §6] |
| P12 | Largest accepted request | Must complete inside the 8 s deadline, enforced by `limits.max_processed_tokens = "auto"` | GATE | [JAS §9.4 #19] |
| P13 | Nightly perf-regression gate | Fails if p50 rises > +5 % or saturated throughput falls > 5 % (median of 3 runs) vs the last release's `reports/perf.json` | GATE | ADR-028 |
| P14 | Laya reference latency, T4, PyTorch fp16 | 39.5 ms for 1 question; ≈ 14.2 ms + 15.1 ms per question (EN); multilingual 32.8 ms | VERIFIED (T4 JSON) / fit ESTIMATED | [BW §1.5] |
| P15 | Jev end-to-end latency | 236–276 ms p50 direct; 800 questions in 985 ms | REPORTED | [JAS §4] |

**Hardware reference, RTX 4090** (REPORTED) [RIS §5.1]:
- 165 TFLOPS dense for bf16/fp16 with fp32 accumulate; 330 for FP8 with fp32 accumulate; 660 TOPS INT8.
- 1,008 GB/s memory bandwidth, 72 MB L2, 128 SMs.

### 5.2 Models and memory

| ID | Quantity | Value | Kind | Evidence |
|---|---|---|---|---|
| MEM1 | `laya-en` / `laya-typed-decisions` parameters | 421,293,830 state-dict elements (421,293,827 `nn.Parameter`s plus the 3-element `temperature` buffer); 842.6 MB as fp16 on disk | VERIFIED (arithmetic); on-disk F16 is VERIFIED by proxy, and the header check is an M0 task | [EA §7.1], [CR G1] |
| MEM2 | `laya-multilingual` parameters | 321,908,998 state-dict elements (321,908,995 `nn.Parameter`s plus the buffer); 643.8 MB (614.0 MiB) as fp16. The "615 MB" file size in the laya-hexagon-npu README matches the MiB value. | VERIFIED (arithmetic) / REPORTED file size | [EA §7.2], [CR G1] |
| MEM3 | All three Laya checkpoints resident on the GPU | ≈ 2.3 GB in bf16 | ESTIMATED | [CR C13] |
| MEM4 | FLOPs per token (L=512) | ModernBERT-large + head ≈ 0.78 GFLOP; mmBERT-base + head ≈ 0.28 GFLOP | ESTIMATED | [RIS §5.2], [EA §7] |
| MEM5 | K/V cache per token, bf16 | 112 KiB (ModernBERT-large: 28 layers × 2 × 1024 × 2 B); 66 KiB (base-size: 22 × 2 × 768 × 2 B) | VERIFIED (arithmetic) | ADR-019 |
| MEM6 | Arena example: GeGLU Wi output at 16,384 tokens | 16,384 × 5,248 × 2 B = 172 MB (164 MiB) | VERIFIED (arithmetic) | [EA §7.3] |
| MEM7 | Laya budgets | EN 512 / 192; multilingual and typed-decisions 1024 / 256 at serve time. typed-decisions was *trained* at 512/192. | VERIFIED | [CR C10, G13] |
| MEM8 | Laya hard option ceiling | ≈ 125 options (EN) / ≈ 250 (1024-token checkpoints); options are cut to 3 text tokens long before that | VERIFIED | [CR G11] |

### 5.3 Serving and wire defaults

| ID | Quantity | Value | Kind | Evidence |
|---|---|---|---|---|
| S1 | Max choice options | 255 (the 256th returns a 400) | VERIFIED Jev contract | [JAS §0] |
| S2 | Max score levels | 10 | REPORTED Jev contract | [JAS §0] |
| S3 | Max request tokens | 65,536 | Jev contract ("64k") | [JAS §0, §13 #6] |
| S4 | Max state + question tokens | 32,768 (Jev accepted a 32,553-token state) | Jev contract ("32k") | [JAS §0], [BW §1.3] |
| S5 | Request deadline `request_timeout_ms` | 8,000 ms. The SDK per-attempt timeout is 10 s (VERIFIED). | Default | [JAS §3.7, §9.4 #19] |
| S6 | `max_body_bytes` | 8,388,608 | Default | ADR-017 |
| S7 | `max_concurrent_requests` | 512 | Default | ADR-017 |
| S8 | `max_batch_tokens` | 16,384 (large); 32,768 (base-size) | Default | [RIS §7] |
| S9 | `max_wait_us` | 0 | Default | ADR-010 |
| S10 | `max_queued_tokens` | 131,072 | Default | ADR-010 |
| S11 | `full` rounding | Floor 1e-6 before renormalisation; the sum equals 1 within 1e-9; no hard zeros | Default | [JAS §9.4 #6], [BW J3] |
| S12 | SDK retry policy | Python: 2 retries, total 30 s; retries 408/429/5xx (incl. 529); honours `retry-after-ms` | VERIFIED | [JAS §3.7] |

### 5.4 Parity tolerances (ADR-014)

| ID | Comparison | Gate |
|---|---|---|
| T1 | Token ids, `build_sequence` ids, markers and errors (≥ 10k items per tokenizer) | 100 % identical |
| T2 | Post-processing given identical logits | Byte-identical JSON |
| T3 | `cpu` fp32 vs PyTorch CPU fp32 (fused fast path) on real weights | max \|Δlogit\| ≤ 1e-4 and max \|Δp\| ≤ 1e-4. Argmax 100 %, except items whose reference top-2 margin is < 1e-3 (listed, not failed). `input_tokens` exact. `round4` JSON identical on ≥ 99.9 % of answers. |
| T4 | GPU bf16 (`candle-cuda` or `cuda`) vs PyTorch CUDA bf16 autocast on the same 4090 | mean \|Δp\| ≤ 2e-3 and max \|Δp\| ≤ 2e-2. Argmax identical wherever the reference top-2 margin is > 0.05; ≥ 99.5 % overall. |
| T5 | Rust CUDA fp32 (the debug-only fp32 GPU path, Q17) vs Rust CPU fp32 | max \|Δp\| ≤ 1e-4 |
| T6 | Tiny random-weight models, CPU fp32 | max \|Δlogit\| ≤ 1e-5 |
| T7 | feishu_zh (multilingual, MPS fp32 archive, 128 requests) | `input_tokens` exact, \|Δp\| ≤ 2e-4, argmax 128/128 |
| T8 | MASSIVE 51-language EN sweep (upstream ran laya 0.2.0 / torch 2.8) | Each per-language accuracy within ±1 item of the 100; macro 0.2269 ± 0.002. Every deviation must be a near-tie (reference margin < 1e-3). |
| T9 | dm2 export parity (Rust vs the PyTorch trainer module) | fp32: \|Δp\| ≤ 1e-4 and argmax 100 %. bf16: max \|Δp\| ≤ 2e-2 and argmax ≥ 99.5 %. Paged-KV attention vs gather-then-dense: ≤ 1e-5 (fp32, on K13's block-table mode, Q17). |
| T10 | fp16 mode admission | ≥ 4× headroom below 65,504 at every GEMM output on the golden set |

### 5.5 Reference quality numbers (for context and comparisons only)

| ID | Quantity | Value | Kind | Evidence |
|---|---|---|---|---|
| Q-ref1 | Laya EN zero-shot on typed-decisions (2,000 decisions) | 0.362, vs random 0.318 and majority 0.461 | VERIFIED | [BW §1.1] |
| Q-ref2 | `laya-typed-decisions` on its test split | 0.766, ECE 0.213 | REPORTED (no raw data) | [BW §1.1] |
| Q-ref3 | verdict2 ModernBERT-base (marker read-out, no head) on typed-decisions | 0.771; correctness-head ECE 0.0144 | REPORTED | [CR G10] |
| Q-ref4 | TF-IDF + LR on typed-decisions | 0.661 | REPORTED | [CR G10] |
| Q-ref5 | JevBench v1.2 hard tier | Jev 74.1 % vs Laya EN 34.1 % | REPORTED | [BW §1.2] |
| Q-ref6 | MASSIVE, 51 languages, Laya EN | macro 0.2269 | VERIFIED (upstream JSON) | [LIS §13 #62] |
| Q-ref7 | Laya shipped ECE, mean over 49 T4 suites | EN 0.466; multilingual 0.314 | VERIFIED | [BW §1.4] |
| Q-ref8 | Jev raw ECE on most tasks | 0.05–0.08 (emotion 0.28–0.35) | REPORTED | [BW §0, §1.4] |
| Q-ref9 | Option-order flip rate | Laya 0.15–0.23 (20-option MASSIVE); Jev 0.13 (DMB S4) and 0.05 (JevBench #40) | VERIFIED / REPORTED | [BW F8, §1.3] |
| Q-ref10 | Banking77 at 77 options | Laya 0.425; Jev 0.763 (DMB S1); Jev 0.870 on the 72-label BTZSC variant | VERIFIED / REPORTED | [BW §1.1, §1.3] |
| Q-ref11 | kotoba (H100) backbone results | DeBERTa-v3-large 0.787 (3k states / 1 epoch) and 0.855 (18k / 1 epoch). ModernBERT-large 0.388–0.399 (3k / 1 epoch, span head). ModernBERT-base 0.539 (3k / 1 epoch) and 0.717 (18k / 2 epochs). | REPORTED | kotoba README:97–187 |
| Q-ref12 | `laya-multilingual` training | Head trained from scratch on mmBERT-base: 15,987 updates, 4 epochs, 4.97 h | VERIFIED (config) | [LTR §0 table] |

### 5.6 dm2 release gates (ADR-022; summarised here, defined there)

| ID | Gate (on the OOD-S test split unless noted) |
|---|---|
| G-Q1 (primary) | Macro accuracy Δ vs `laya-en` ≥ +10 pp, with the paired-bootstrap CI lower bound > +5 pp. NLL and Brier better, with CIs excluding 0. |
| G-Q2 (reported, not blocking) | Recover ≥ 50 % of the (Jev − Laya) accuracy gap, macro-averaged over the pool-J suites, using third-party published Jev numbers only |
| G-Q3 | Distribution ECE-15 ≤ 0.05 after an in-domain fit; `p_correct` ECE ≤ 0.03; no temperature at a clamp bound |
| G-Q4 | Coverage at ≤ 5 % error ≥ 0.60; "unknowable" items answered at ≥ 0.9 confidence in ≤ 5 % of cases |
| G-Q5 | Flip rate ≤ 0.05 on the 20-option MASSIVE-en permutation probe; noul label-swap consistency ≥ 0.90; shuffled-context accuracy within 2 pp of the label prior; question isolation bitwise under layout L2 |
| G-Q6 | Code-word test (DMB S3 protocol) ≥ 0.98 at k = 255; Banking77-77 (held-out source) ≥ `laya-en` + 20 pp (goal ≥ 0.75); chunk invariance max \|Δp\| ≤ 0.02 |

### 5.7 Training and compute

| ID | Quantity | Value | Kind | Evidence |
|---|---|---|---|---|
| C1 | Trainer throughput, ModernBERT-large, packed varlen + compile | 20–30k tok/s planning figure; base-size models ≈ 2.4× that. M0 exit gate: ≥ 20k tok/s, otherwise re-plan. | ESTIMATED / GATE | [CR C7] |
| C2 | Training memory | 6.3–7.0 GiB static; 1.0–1.2 MiB per token of activations; ≈ 12k tokens per micro-batch without checkpointing | ESTIMATED | [CR C13], [RT §4] |
| C3 | Micro-batch and step | 12,288 tokens per micro-batch (static) × accumulation 4 ≈ 49k tokens per optimiser step | Default | ADR-023 |
| C4 | Teacher labelling | ≈ 300M prefill tokens per 1M decisions per teacher, ≈ 10–17 GPU-h | ESTIMATED | [RT §6.2] |
| C5 | GPU budget to v0.3 | 126–193 GPU-h; 158–241 GPU-h with 25 % retry slack (derived in ADR-026) | ESTIMATED | ADR-026 |
| C6 | Power limit during training | 350–380 W (costs about 5–10 % throughput) | ESTIMATED | [RT §5.1] |

### 5.8 Roadmap (ADR-032; 12 h/week, no extra buffer)

| ID | Release | Cumulative dev-h | Week | ≈ Month |
|---|---|---|---|---|
| R-v0.1 | v0.1 "drop-in" | 158 | 14 | 3.2 |
| R-v0.2 | v0.2 "engine" | 364 | 31 | 7.1 |
| R-v0.3 | v0.3 "first own model" | 505 | 43 | 9.9 |
| R-1.0 | 1.0 "API freeze" | 535 | 45 | 10.4 |
| R-1.1 | 1.1 multilingual (only if Q8 clears) | 575 | 48 | 11.0 |
| R-1.2 | 1.2 FP8 opt-in | 605 | 51 | 11.7 |

Time-budget sensitivity: at 10 h/week, multiply the week numbers by 1.2; at 8 h/week, by 1.5.

## 6. Resolved disagreements (index)

The names, glossary and numbers in §3–§5 already reflect these resolutions.

| # | Disagreement | Resolution | ADR |
|---|---|---|---|
| D1 | When to build the custom CUDA engine: perf in 0.1, accuracy at M5 (weeks 22–32), product in v0.2 | v0.1 ships on candle CPU plus candle-CUDA. The custom engine starts right after v0.1 with the laya-v1 kernel subset, which Laya parity needs anyway and which is a strict subset of what dm2 needs. Only the paged/shared-state path waits for the layout ablation. | ADR-006, ADR-032 |
| D2 | dm2 layout: perf's isolated-option tree, accuracy's L0/L2/L3, product's shared state with a top-k fallback | Accuracy's L0/L2/L3 with its pre-registered rule; perf's tree is one extra ablation arm only | ADR-019 |
| D3 | Backbone default and adoption rule | Early A/B (E1, weeks 5–11) plus a final X1. DeBERTa wins only by > 3 pp, and then as a teacher distilled into a ModernBERT-family student. Perf's "tree must be expressible" condition is dropped. | ADR-020 |
| D4 | Default numeric output: full precision (perf, accuracy) vs 4 decimals with a 1e-4 floor (product); the review panels split | Full precision with a 1e-6 floor. The JAS §9.4 #6 SHOULD and the "optimal accuracy" goal decide it. `round2`/`round4` are opt-in. | ADR-016 |
| D5 | Laya GPU precision: perf's `auto` → fp16 vs bf16 | bf16 default [CR C1]; fp16 opt-in after the sweep, with a runtime non-finite fallback | ADR-009 |
| D6 | Batching wait: product's 1,500 µs vs zero-delay | Zero-delay (`max_wait_us = 0`) | ADR-010 |
| D7 | Training data vs the Jev-comparable suites | Accuracy's exclusion list plus three disjoint pools. MASSIVE becomes eval-only, which fixes accuracy's own train/held-out contradiction. | ADR-024 |
| D8 | P1 typed-decisions reproduction: run by default (perf, product) vs gated (accuracy) | The maintainer's opt-in (Q3). Default: licence-clean trainer validation. | ADR-023 |
| D9 | Deadline and overload codes | 8 s deadline; 503 (529 in `strict`) for overload and deadline; 429 only per key | ADR-017 |
| D10 | Naming: `tydec-*`, `rustify-*`, Arbitro | Arbitro (working name, confirmed by the maintainer on 2026-09-24, Q1). Modes and methods carry no third-party marks. | ADR-002, ADR-031 |
| D11 | Crate count: 14, 18, or 8 published + 3 internal | 8 published + 3 internal; the other proposals' boundaries become modules | ADR-003 |
| D12 | Router, email and presets in v0.1 (perf, accuracy) vs deferred (product) | v0.1 routing is `explicit`. The laya-heuristic router lands in v0.2. Email, presets and shortlist are community work, self-checked by L0 fixtures. | ADR-012 |
| D13 | Score confidence: peak (critic, empirical) vs the documented formula (JAS §9.4 #7) | `peak` by default in `strict` and `lenient`, configurable; documented as a deliberate choice | ADR-016 |
| D14 | Many options: 32-token spans under a 4,096 cap (product) vs 64-token branches (perf) vs chunking (accuracy) | Option-group chunking (≤ 64 options per group) plus random chunking during training | ADR-019 |

---

## 7. Architecture Decision Records

Each ADR follows the same structure: Status, Area, Context, Decision, Alternatives, Consequences, Validation gate, Evidence. The status is one of:
- **Accepted**;
- **Proposed-needs-user-input** (a Q in §2);
- **Deferred-until-measured** (it names the measurement that settles it).

Downstream `docs/adr/ADR-NNN-<slug>.md` files are split from this section verbatim.

---

### ADR-001: Product scope, two tracks, non-goals, definition of done

**Status:** Accepted. **Area:** A1.

**Context.**
- Jev is closed and hosted. Laya has open weights, but its zero-shot accuracy is weak (Q-ref1, Q-ref5), its Python serving is slow (P14), and its wire compatibility is loose [JAS §10].
- The research says parity and model improvement are separate goals and must be measured separately [BW §5.1].
- Prior-art Rust crates exist (`laya` 0.1.1 and `laya-rs` 0.1.0: candle, f32, no Jev contract, no parity goldens).
- The maintainer is one person working about 12 h/week.

**Decision.**

*Two tracks*, shipped from one workspace and never mixed in one results table (ADR-027):
- **Compat runtime** (track A). It runs the user-downloaded checkpoints `laya-en`, `laya-multilingual` and `laya-typed-decisions` with verified parity (ADR-014), on a fast engine (ADR-006, ADR-008), behind a Jev-compatible server (ADR-015).
- **Own-model track** (track B). The `dm2` family (`arbitro-en-large`, `arbitro-en-base`, and `arbitro-multi-base` after Q8), with Apache-2.0 weights trained on the 4090.

*laya-plus* options (`arbitro calibrate`, `x_arbitro.permutations`, the noul empty-state prior correction) are opt-in, are reported in their own rows, and are never called parity.

*Surfaces:*
- the `arbitro` binary;
- the Rust facade crate `arbitro`;
- Docker images `-cpu` and `-cuda`;
- the PyPI wheel `arbitro`: CPU inference plus the data/calibration/eval core, public from v0.3. The trainer uses it internally from M3a.

*Non-goals* (these also go into the README section "What this is not"):
- text generation or chat, and embeddings as a product;
- a decoder "accurate tier" before 1.0 (Q12);
- a hosted service with accounts or billing;
- emulating Jev's randomness (probability-key shuffling, ±0.04 noul noise);
- any use of Jev outputs, or of a TypeSafe account;
- redistributing Laya weights, initialising from them, or distilling from Laya outputs;
- a pure-Rust training loop before 1.0;
- multi-GPU, ROCm or Intel GPUs;
- CUDA wheels on PyPI in 0.x;
- native 32k encoder context. States longer than 8k tokens are truncated head+tail, and the truncation is reported.

*Releases:*
- v0.1 "drop-in";
- v0.2 "engine" (the custom CUDA engine and the laya-heuristic router);
- v0.3 "first own model";
- 1.0 "API freeze" (`/v1` and `x_arbitro` v1 frozen; semver checks).

*v0.1 definition of done* (all must hold):
1. `docker run -p 8080:8080 -v arbitro-cache:/cache ghcr.io/foxur/arbitro:0.1.0-cpu serve --preload laya-en` downloads the pinned checkpoint on first start (printing its licence line) and serves requests. The image sets the bind address and `ARBITRO_HOME=/cache` (ADR-017).
2. The unmodified `typesafe-sdk==0.7.1` quickstart (`examples/sdk_quickstart.py`) succeeds with only `TYPESAFE_BASE_URL` and a local `TYPESAFE_API_KEY` changed.
3. The SDK conformance suite passes 100 % (ADR-015).
4. Parity levels L0–L3 are green for all three checkpoints on `cpu` fp32 and `candle-cuda` bf16 (T1–T6; T5 on the debug fp32 path, Q17).
5. Gates P6 and P7 are met.
6. The README states the non-goals and the non-affiliation disclaimer.

*Performance goals* (P1–P5) are goals. Release gates are the separate, looser values in §5.1.

**Alternatives.**
- Model-first: rejected. It ships nothing for months, and the model is the riskiest bet [CR C9, G5].
- Building on `laya` / `laya-rs`: partly adopted. We offer `pycompat` upstream, but those crates are young, f32-only, have no Jev contract, and our engine would be a rewrite anyway.
- A Python server with Rust kernels: rejected [BW §2.4].
- Forking TEI: rejected. It pins a 0.8-era candle, keeps an fp16 residual, uses tanh GELU and has no head [RIS §3.1–3.2]. We reuse its architecture pattern only.
- The custom engine in v0.1 (perf): rejected on hours (ADR-032).
- A decoder tier in the own-model family (accuracy): rejected before 1.0.

**Consequences.**
- The first usable release arrives at about week 14. The performance headline arrives with v0.2 at about week 31.
- The runtime is useful on day one: a Laya checkpoint behind the Jev wire contract on a local GPU at an estimated 6–10 ms per question (P6), against Jev's 236–276 ms network latency (P15).
- The model quality claims wait for v0.3.

**Validation gate.** The v0.1 definition of done above, and each release's gates in ADR-032.

**Evidence.** [BW §0, §1.5, §2.4, §5.1]; [JAS §3.7, §9, §10]; [RIS §3.1–3.2, §5.3]; [CR C6, C9, G5]; [LTR §5.2]; review panels (A1).

---

### ADR-002: Project and package naming

**Status:** Accepted 2026-09-24 (Q1 answered: Arbitro). The trademark search is still to be recorded before the first `cargo publish`. **Area:** A2.

**Context.**
- `rustify` on crates.io is an unrelated HTTP-client crate (0.7.0, about 12.4M downloads), and `rustify-cli` is taken. npm `rustify` is taken; PyPI `rustify` is free.
- The design phase re-checked on 2026-09-23: `arbitro` and every planned `arbitro-*` crate name are free on crates.io (index 404), and `arbitro` is free on PyPI and npm.
- `tydec` is free on crates.io and PyPI but taken on npm.
- `laya` and `laya-rs` are taken (prior art).

**Decision.**
- The name is **Arbitro** (maintainer decision, 2026-09-24). §3 lists every identifier derived from it; the rename rule stays as a contingency if the trademark search fails.
- No `cargo publish` happens before a trademark search (UNVERIFIED so far) is recorded in `docs/adr/`.
- Names are claimed only by publishing a real 0.0.1 of `arbitro-proto`, because crates.io forbids squatting.
- The repo stays `Foxur/Rustify`; GitHub redirects after a rename.
- No crate, binary, image, mode, method or config key carries "Jev", "TypeSafe" or "System One". Wire-required identifiers are exempt (ADR-031).
- Registry ids for third-party checkpoints (`laya-*`), the `laya` mode and the `laya-v1` family are nominative compatibility descriptors. No crate is named `laya*`.

**Alternatives.**
- `rustify-*`: its crates would sit next to a popular unrelated crate and be confused with it.
- `tydec`: taken on npm and hard to say out loud.
- `verdikt`, `decidr`, `noul`: `noul` collides with a question-type term, and the others were not checked on npm.

**Consequences.** A rename costs one mechanical find-and-replace, as long as it happens before the first publish.

**Validation gate.** At decision time: re-query crates.io, PyPI and npm, record a trademark search, and have the maintainer sign off.

**Evidence.** Registry checks during the design phase (2026-09-23); [P-prod A2], [P-perf A2], [P-acc A2.1]; review panels (A2).

---

### ADR-003: Workspace layout, crate boundaries, toolchain

**Status:** Accepted. **Area:** A2.

**Context.** Every published crate carries semver, docs.rs and changelog overhead. The proposals ranged from 8 to 18 crates. One maintainer can sustain about 8.

**Decision.**

*Eight published crates, versioned in lockstep:*

| Crate | Responsibility (main modules) |
|---|---|
| `arbitro-proto` | Wire types, pydantic-style 422 builder, error bodies, OpenAPI. serde only. |
| `arbitro-core` | Domain types; tokenizer wrapper (`tokenizers =0.23.2`); `registry` + loader (safetensors mmap, both ModernBERT config formats); the `Backend`/`Runner`/`Frontend` traits; `PackedBatch`; `family::dm2` (layout planner, option chunking, none option); calibration *application*; shared post-processing. Never depends on a GPU crate. |
| `arbitro-compat` | Laya 0.3.7 behaviour. `pycompat` (json, fmt, unicode, npsum); `render`/`validate`/`sequence`/`temps`/`post`; the laya-v1 frontend; `lang`/`router` (v0.2); `email`/`shortlist` (community). |
| `arbitro-candle` | candle 0.11 backends `cpu`, `metal` and `candle-cuda`, with our own ModernBERT and head code. |
| `arbitro-cuda` | The custom engine (ADR-008). Behind a feature; needs nvcc. |
| `arbitro-server` | axum routes, modes, scheduler (TEI-derived), auth, limits, metrics, config. |
| `arbitro-eval` | Metrics, suites, probes, statistics, gates, calibration *fitting* (`calib_fit`), claims check. |
| `arbitro` | Facade library (`Engine`) and the CLI binary. It forwards the features `cuda`, `candle-cuda`, `metal`, `mkl` and `accelerate`. |

*Internal crates* (`publish = false`):
- `arbitro-data`: dataset converters, augmentation, packer, manifests, licence gate, MinHash/13-gram checks;
- `arbitro-py`: the PyO3 abi3 wheel;
- `xtask`.

*Dependency direction:* `proto ← core ← {compat, candle, cuda, eval} ← server ← arbitro`. `arbitro-data` depends on `core`, `compat` and `eval`. `arbitro-py` depends on `arbitro` and `arbitro-data`.

*Toolchain and dependencies:*
- Edition 2024.
- `rust-toolchain.toml` pins **1.98.1**, the current stable (VERIFIED 2026-09-23 via static.rust-lang.org). The **MSRV is 1.96** (pinned minus 2) and is tested in CI. The toolchain is bumped deliberately per release. Documents must not call "1.94" the toolchain.
- Key dependencies:
  - `tokenizers =0.23.2`, because 1.0.0-rc.2 refuses the shipped files [EA §8.4];
  - `candle-* 0.11`;
  - `cudarc 0.19.9`;
  - `serde_json` with `preserve_order` and `arbitrary_precision`;
  - `indexmap`, `axum`/`tokio`, `pyo3`/`maturin`.

*Release tooling:* release-plz; `cargo-semver-checks` from v0.2; `cargo-deny`.

*Repository layout:* §3.6.

**Alternatives.**
- 14 crates (perf) or 18 (accuracy): rejected on maintenance cost. Their boundaries survive as modules.
- A monolith: rejected, because CUDA and axum would leak into library users.

**Consequences.** Lockstep releases republish unchanged crates, which is an acceptable cost. `pycompat` can later be extracted as its own crate if upstream wants it.

**Validation gate.** In CI:
- `arbitro-core` builds without GPU features;
- `cargo build -p arbitro --no-default-features` succeeds;
- the MSRV job passes;
- `cargo-deny` passes;
- `cargo-semver-checks` passes (from v0.2).

**Evidence.** [P-prod A2], [P-perf A2], [P-acc A2.2]; [EA §8.4]; [RIS §1]; review panels (A2).

---

### ADR-004: Public Rust API and core abstractions

**Status:** Accepted. **Area:** A2.

**Context.**
- The GPU runner is a blocking object owned by one thread, as in TEI [RIS §3.3].
- The dm2 layouts need explicit positions and different query and key segment lengths. Product's `prefix: Option<SharedPrefix>` cannot express layout L3, option chunking or positions that continue after the state.

**Decision.** The sketch below is normative for names and field semantics.

```rust
// arbitro-proto: key order preserved with IndexMap
pub struct DecideRequest {
    pub state: Option<serde_json::Value>,             // string | object | array; null only in `laya` mode
    pub model: Option<String>,                        // required in `strict`
    pub questions: IndexMap<String, Question>,
    pub x_arbitro: Option<RequestExt>,
    #[serde(flatten)] pub extra: serde_json::Map<String, serde_json::Value>, // accepted and ignored
}
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Question {
    Choice { instructions: Option<Value>, criteria: ChoiceCriteria },
    Score  { instructions: Option<Value>, criteria: Vec<Value> },
    Noul   { instructions: Option<Value>, criteria: Option<NoulCriteria> },
}
#[serde(untagged)] pub enum ChoiceCriteria { Map(IndexMap<String, Option<Value>>), List(Vec<String>) } // List: lenient/laya only
pub struct DecideResponse { pub model: String, pub answers: IndexMap<String, Answer>, pub usage: Usage,
                            pub x_arbitro: Option<ResponseExt> }

// arbitro-core
pub trait Backend: Send + Sync + 'static {
    fn id(&self) -> &'static str;                       // "cpu" | "cuda" | "candle-cuda" | "metal"
    fn load(&self, m: Arc<ModelArtifacts>, o: &RunnerOptions) -> Result<Box<dyn Runner>>;
}
pub trait Runner: Send {                                  // owned by one backend thread; not async
    fn caps(&self) -> RunnerCaps;                         // precisions, max_batch_tokens, layouts, deterministic
    fn warmup(&mut self) -> Result<()>;                   // also captures CUDA graphs; /ready flips after this
    fn submit(&mut self, b: PackedBatch) -> Result<Ticket>; // one batch in flight while the next is packed
    fn wait(&mut self, t: Ticket) -> Result<RawOutputs>;
}
pub trait Frontend: Send + Sync {                         // one per family: laya-v1 (in arbitro-compat), dm2 (in core)
    fn plan(&self, req: &ValidatedRequest, tok: &TokenCache) -> Result<WorkPlan>;
    fn finish(&self, plan: &WorkPlan, raw: &RawOutputs, cal: &Calibration, out: &OutputOptions)
        -> Result<IndexMap<String, Answer>>;
}
pub struct PackedBatch {
    pub input_ids: Vec<u32>,
    pub position_ids: Vec<u32>,           // explicit; under layout L2 they continue after the state
    pub cu_seqlens_q: Vec<u32>,
    pub cu_seqlens_k: Vec<u32>,           // equals cu_seqlens_q for laya-v1 and layout L0
    pub max_seqlen_q: u32, pub max_seqlen_k: u32,
    pub kv_segments: Vec<KvSegment>,      // per query sequence: which state K/V (cached or in-batch) it reads
    pub qtype: Vec<u8>,                   // per sequence
    pub cu_markers: Vec<u32>, pub marker_rows: Vec<u32>,
    pub span_ranges: Vec<(u32, u32)>,     // dm2 option spans; empty for laya-v1
    pub want_act: bool,                   // laya-v1 act head (laya mode or on request)
}
pub struct RawOutputs { pub logits: Vec<f32>, pub act_logits: Option<Vec<f32>>, pub aux: Option<Vec<f32>> }
pub enum Precision { F32, Bf16, Fp16Checked, Fp8W8A8 }  // Fp8: per-channel weights, per-token activations

// arbitro (facade)
let engine = arbitro::Engine::builder().model("laya-en").device(Device::Auto).build()?;
let resp = engine.decide(req).await?;            // or engine.decide_blocking(req)
```

Python mirrors this: `arbitro.Engine(model="laya-en").decide(state, questions)`.

**Alternatives.**
- An async-trait runner: rejected, because a GPU runner is blocking.
- Perf's `TreeLayout` field: rejected, because `kv_segments` plus `position_ids` express the tree arm too.
- Product's `SharedPrefix`: rejected as too narrow.

**Consequences.** v0.1 fills only the laya-v1 fields. dm2 adds no breaking change. `Precision::Fp16Checked` encodes the fp16 overflow-sweep gate in the type system.

**Validation gate.**
- Doc tests for every public example.
- An API snapshot diff on each PR.
- A laya-v1 `PackedBatch` round-trip test.
- The paged-KV ≡ gather-then-dense test (T9).

**Evidence.** [RIS §3.3, §9]; [CR G5]; [P-acc A2.3], [P-perf A2], [P-prod A2]; review panels (A2).

---

### ADR-005: Model registry and checkpoint pinning

**Status:** Accepted. The per-file sha256 values are filled in during M0. **Area:** A2.

**Context.**
- Hub revisions drift. The EN root benchmarked by laya-mps is `c5d78730`. The mirrored bundle is `1c5edc17`, and it has no `typed-decisions` folder. typed-decisions is a separate repo, pinned at `f9ab0b22` [CR G1, C10, §3.9].
- One bundle SHA cannot pin all three checkpoints.

**Decision.**

*Registry format.* A built-in `registry.toml`, extendable via `~/.config/arbitro/models.toml`. Fields:
- `id`, `aliases`, `family` (`laya-v1` | `dm2`);
- `source`: `{hf, subfolder, revision (40 hex)}`, `{dir}` or `{url}`;
- `files`: path → sha256;
- `license`, `redistribute` (bool);
- budgets (`max_len`, `head_max_len` for laya-v1);
- `precision_allow`, `calibration`.

*Pins:* per checkpoint, per file (§3.5). Each golden fixture header records the revision and sha256 it was produced with.

*Auto-detection:*
- a local directory containing `rl_agent_config.json` is `laya-v1`;
- one containing `arbitro-model.json` is `dm2`.

*Loading:*
- a strict key set: 206 tensors for EN and typed-decisions, 170 for multilingual [EA §9.1];
- the dtype is read per tensor (F16/F32/BF16);
- the `temperature` buffer is loaded and ignored [CR C3];
- runtime temperatures come from `rl_agent_config.json`;
- `_fix_tokenizer_config` is applied in memory only;
- special-token ids come from the tokenizer files, never from `encoder/config.json`. The multilingual `cls_token_id=1` is wrong [CR G2].

*`arbitro pull`:*
- downloads to `$ARBITRO_HOME`, reusing the HF cache via `hf-hub`;
- verifies sha256;
- prints the licence line;
- never mirrors Laya weights anywhere.

*Residency:* all three Laya checkpoints stay resident (≈ 2.3 GB, MEM3), with `models.max_resident = 3` and LRU eviction beyond that. This removes Laya's 7–10 s reload stalls.

*Responses* carry the concrete id (§3.5), never an alias, in `strict` and `lenient` mode. `laya` mode returns laya-serve's `"laya-rl-agent"` for byte parity (ADR-015); the `x-arbitro-model` header carries the concrete id in every mode.

*M0 tasks:*
- read the safetensors headers and confirm F16 [CR G1];
- record the sha256 of every file;
- fetch EN root weights at both `c5d78730` and `1c5edc17` and compare their sha256, which settles [CR §3 #9];
- re-read `amp_dtype` in the configs at the `c5d78730` and `f9ab0b22` pins ([CR C1] read it at `1c5edc17`);
- record everything in `registry.toml` and `docs/perf-baseline.md`.

**Alternatives.** Product's single `1c5edc17` pin: rejected, because it is wrong for EN and typed-decisions.

**Consequences.** Adding a new Laya revision is a registry edit plus fixture regeneration.

**Validation gate.**
- `arbitro pull` refuses a sha256 mismatch.
- The L3 fixtures record the pins.
- A CI test checks that a registry entry without `sha256` fails to load.

**Evidence.** [CR G1, G2, C3, C10, §3 #9]; [LIS §6, §13 #61]; [EA §9.1]; `afshinm/laya-mps/src/laya_mps/laya_setup.py`; `EricYu123456/laya-hexagon-npu/README.md:248`.

---

### ADR-006: Backend tiers and engine sequencing

**Status:** Accepted. **Area:** A3.

**Context.**
- No off-the-shelf Rust runtime runs Laya well. candle-transformers' ModernBERT is f32-only, builds dense masks and computes RoPE tables in half precision. TEI is fp16-only, uses tanh GELU and is pinned to old candle and cudarc versions [RIS §0, §2.1, §3].
- The custom engine is 3–5k lines of Rust plus 1–2k lines of CUDA [RIS §8]; the review panels estimate 150–200 h.
- The laya-v1 kernel set is needed for Laya parity regardless of any dm2 decision, and it is a strict subset of what dm2 layout L2 needs [P-acc A10].

**Decision.**

| Tier | Backend | Crate | Precision | From | Status |
|---|---|---|---|---|---|
| 1 | `cpu` | arbitro-candle | fp32 (golden) | v0.1 | Release-blocking; every PR |
| 1 → 2 | `candle-cuda` | arbitro-candle | bf16 via candle-flash-attn varlen+window; fp32 debug-only via unfused masked attention (T5, Q17) | v0.1 | Default GPU backend in v0.1; a debug fallback from v0.2 |
| 1 | `cuda` | arbitro-cuda | bf16 default; fp16 opt-in; fp8 in 1.x; fp32 debug-only via K13 (T5, T9, Q17) | v0.2 | Release-blocking from v0.2; nightly on the 4090 |
| 2 | `metal` | arbitro-candle | f32 (f16 later) | v0.1 (best effort) | macOS CI builds plus CPU tests |
| 3 | `ort` | `arbitro-ort` (reserved) | fp32 | only if ADR-007 fails | — |

*Sequencing:*
1. The custom engine (M4) starts right after v0.1 and M3a, with the **laya-v1 kernel subset**: dense varlen FA2, fused LayerNorm/GeGLU/RoPE, the head and the act head.
2. The paged split-KV path for dm2 layout L2 is built only after the M5 layout decision, inside M6.
3. `candle-cuda` stays the default GPU backend until `cuda` passes T4 and gates P1–P4.

*Fallback (cut rule C-3, ADR-033):* if M4 exceeds 255 dev-h (150 % of budget):
- `candle-cuda` remains the default;
- the compat completion work (M4c) ships as v0.1.x;
- the engine continues into a later v0.2.

Both GPU paths use our own ModernBERT code. We never use candle-transformers' model.

**Alternatives.**
- The custom engine in v0.1 (perf): its own hours make a 16-week 0.1 infeasible.
- Deferring the engine to M5 after the dm2 freeze (accuracy): there is no technical dependency, and it costs about 6 months of the performance headline.
- TEI fork, ORT/TensorRT as primary, burn: rejected [RIS §3–4].

**Consequences.**
- v0.1 GPU latency is launch-bound: about 8–10 ms at L=512 [RIS §5.3].
- v0.2 brings the headline numbers.
- The dm2 kernels reuse the v0.2 kernels.

**Validation gate.**
- v0.1: P6 and T4 on `candle-cuda`.
- v0.2: P1–P4, T4 and T5 on `cuda`, plus the bitwise batch-invariance test (ADR-011).

**Evidence.** [RIS §0, §2, §3, §4, §5.3, §8]; [CR C6]; [P-prod A3], [P-perf A3], [P-acc A3, A10]; review panels (A3, critical disagreement 1).

---

### ADR-007: CPU backend decision tree

**Status:** Deferred-until-measured. The M0 CPU spike settles it in week 2. **Area:** A3.

**Context.** candle's default CPU GEMM measured 62–70 GFLOP/s, against 232–246 GFLOP/s for PyTorch CPU, i.e. 3.3–3.9× slower. That was VERIFIED on a 4-core Xeon. MKL could not be tested in the sandbox [RIS §2.6].

**Decision.**
- The spike measures candle (default gemm), candle+MKL (x86) or Accelerate (macOS), and PyTorch CPU fp32. It uses ModernBERT-large layer shapes at L = 256 and 512, on the maintainer's CPU, with 8 physical cores as the target.
- If candle+MKL/Accelerate is ≤ 1.5× PyTorch, candle stays Tier-1 CPU. MKL is linked dynamically and never statically without counsel (ADR-030).
- Otherwise `arbitro-ort` becomes Tier-1 CPU. It runs an ONNX graph *generated in Rust from safetensors* (`arbitro export-onnx`, no Python), with k padded to ≥ 2 to avoid the `topk(2)` trap [RIS §4.2].
- In both cases candle `cpu` fp32 remains the parity reference (T3).
- `arbitro-en-base` is the CPU-friendly dm2 model.

**Alternatives.** ORT first: rejected, because it is a heavy dependency, needs a generated graph and has padding traps.

**Consequences.** One extra crate, but only if the spike fails. ADR-032 budgets no dev-h for `arbitro-ort`: if the spike fails, the M0 exit review estimates the ORT path, and cut rule C-1 applies to M1.

**Validation gate.** P7 on the release CPU. The spike report goes to `reports/spikes/cpu.md`.

**Evidence.** [RIS §2.6, §4.2]; [CR G12]; [P-prod A3].

---

### ADR-008: Custom CUDA engine design (`arbitro-cuda`)

**Status:** Accepted. Items marked "M0 probe" or "M4 measure" are Deferred-until-measured. **Area:** A3.

**Context.**
- The model is fixed and small, so about 12 kernel types cover it.
- Bit-level control over precision, graphs and epilogues is the reason to write our own engine [RIS §8].
- Several cuBLASLt features are unverified on sm_89 [CR G6].

**Decision.**

*Kernels:*

| # | Kernel | Specification |
|---|---|---|
| K1 | `embed_gather_ln` | Token gather + LayerNorm without bias, two-pass fp32 statistics. Writes the fp32 residual and a bf16 GEMM input. |
| K2 | `rope_qk_inplace` | cos/sin tables built in f64 and stored as f32, per θ: EN global 160k / local 10k; multilingual 160k / 160k [EA §2.1]. Gathered by packed `position_ids`; rotate-half. |
| K3 | `fa2_varlen_fwd_hdim64` | Vendored FA2, dense non-split kernels `flash_fwd_hdim64_{bf16,fp16}_sm80.cu`. Window (64,64) on local layers, (−1,−1) on global layers; this equals HF's \|i−j\| ≤ 64 [CR G3]. `num_splits = 1`. The dispatch is trimmed to non-causal hdim64, and the change is recorded in `MODIFICATIONS.md`. |
| K3b | `fa2_paged_splitkv_hdim64` | Only for dm2 layout L2 (M6). Vendors `flash_fwd_splitkv_hdim64_{bf16,fp16}_sm80.cu`, because `flash_api.cu` routes every `block_table` call to `run_mha_fwd_splitkv_paged_` (VERIFIED). `num_splits = 1`; page size a multiple of 32. |
| K4 | `add_ln_nobias` | Residual add + LayerNorm, fp32 statistics, bf16 out. Used whenever the GEMM epilogue did not fuse the add. |
| K5 | `geglu_erf` | Exact-erf GELU(u₁)·u₂ in fp32, bf16 out. Never the tanh epilogue. |
| K6 | `final_ln_type_headln` | `final_norm` + `type_emb[qtype]` on every row + head `norm1` (with bias), fused. |
| K7 | `ln_bias` | Head `norm2` (both layers), head layer 2's `norm1` (K6 covers only layer 1's `norm1`), and `scorer.0`. |
| K8 | `gather_rows` | CLS and marker rows. Enables the exact pruning of head layer 2: Q and FFN only for CLS and marker rows, K/V for all rows. |
| K9 | `scorer_tail` | GELU-erf and the D→1 projection in fp32 over Σk rows. |
| K10 | `act_head` | Runs on the GPU. Features come from the *untempered* fp32 softmax: top1, margin, entropy (1e-9 floor), k/255 with k clamped ≥ 2, top2 = 0 when k = 1 [LIS §5.3, CR C2]. Then the CLS row after the head layers goes through Linear(D+4 → 256) → exact-erf GELU → Linear(256 → 2) in fp32 (`nn.GELU()`, `common.py:101`, VERIFIED). Runs only in `laya` mode or on request. |
| K11 | `absmax_probe` | Per-GEMM-output max-abs, for `sweep-overflow` and debug builds only. |
| K12 | `quant_fp8_rowwise` | Per-token E4M3 quantisation (1.x, ADR-009). |
| K13 | `attn_varlen_f32_ref` | Debug-only (Q17): a naive fp32 varlen attention with the same window semantics as K3, used with fp32 variants of K1 and K4–K7 for T5. A block-table mode (M6) serves T9's paged-vs-dense clause. Never a serving path and never graph-captured. |

*GEMMs:*
- cuBLASLt through the raw `cudarc::cublaslt::sys` API; the safe `Matmul<T>` uses one T for A, B and C [CR G6].
- Inputs bf16 (or fp16), compute fp32 only. Never FP16 accumulate.
- Wqkv, Wi, in_proj, linear1 and scorer.1 write bf16.
- Wo, Wo_mlp, out_proj and linear2 write **fp32 C/D with beta = 1**, which fuses the residual add. This needs an M0 probe on sm_89; if it fails, K4 folds the add in.
- Head linear1 uses the exact `RELU_BIAS` epilogue.

*Build:*
- nvcc produces a fatbin with sm_80/86/89/90 SASS plus compute_90 PTX. Running on sm_120 via PTX is UNVERIFIED.
- CUTLASS is vendored at a pinned tag, with no build-time fetch. candle-flash-attn's build.rs fetches CUTLASS, so we do not use its build.
- Binaries and images are prebuilt.

*CUDA graphs:*
- Phase A: piecewise graphs per token bucket T ∈ {256, 512, 1k, 2k, 4k, 8k, 16k}. GEMMs and element-wise kernels are captured; FA2 launches eagerly (28 + 2 launches).
- Phase B: whole-forward graphs. The mechanism is either bucketed (n_seq, max_seqlen) grids relying on FA2's early exit, or a tile-list FA2. M4 measures both and picks one.

*Memory and I/O:*
- A static arena per model, sized for `max_batch_tokens` (example: MEM6).
- Pinned, double-buffered H2D on a copy stream.
- D2H copies only the Σk logits, plus act logits when requested.

*Exact savers* (outputs unchanged):
- varlen packing without padding;
- in-batch deduplication of identical sequences;
- an answer cache keyed by (concrete model id, precision, calibration id, token ids, qtype), enabled **only** under `batch_invariant`;
- head-layer-2 pruning;
- the act head only when needed;
- the state tokenised once per request (ADR-010).

*Numerics note (corrects [P-perf A3.2/A3.3]):*
- Our bf16 path keeps the residual C/D, GELU and scorer in fp32. That is *closer to fp32* than the PyTorch CUDA reference, not bit-closest to it: the reference rounds every GEMM output, scorer.3 included, to bf16 and runs GELU in bf16 [EA §6.1].
- T4 therefore budgets for the reference's own logit quantisation (a step of about 0.03 at |z| ≈ 5).
- The nightly job also *reports* bf16 vs the fp32 golden, where we expect to be closer.

*M0 probes, recorded in `reports/spikes/cuda.md`:*
- the cuBLASLt bf16 → fp32 C/D with beta = 1 on sm_89, and its TFLOPS;
- FA2 hdim64 windowed varlen throughput;
- GEMM TFLOPS for bf16 / fp16-with-fp32-accumulate / FP8 at the model's (N, K) and M ∈ {256, 1k, 4k, 16k};
- the FP8 `MATRIX_SCALE_OUTER_VEC_32F` probe, for information only (ADR-009).

*Budget:* 170 dev-h (M4, ADR-032).

**Alternatives.**
- candle-CUDA with a TEI-style flash path as the primary engine: launch-bound, about 500 ops per forward [RIS §2.2].
- ORT/TensorRT: dense window masks, padding, multi-GB dependencies [RIS §4.2–4.3].

**Consequences.**
- About 3–5k lines of Rust and 1–2k of CUDA to maintain.
- Vendored third-party code needs provenance records (ADR-030).

**Validation gate.**
- Per-kernel unit tests against float64 numpy on tiny shapes.
- A layer-by-layer `--dump-activations` comparison against PyTorch hooks: embedding LN, layer 0 (global), layer 1 (local), `final_norm`, head, logits and act logits, with ragged lengths > 129.
- T4 and T5.
- Microbenchmarks per kernel (TFLOP/s, GB/s) stored as regression baselines.
- P1–P4.

**Evidence.** [RIS §2.1–2.3, §4.1, §5.3, §6, §7, §8]; [EA §2.1, §3.2, §6.1–6.2, §7.3]; [CR G3, G6, C2]; [LIS §5.3]; `candle-flash-attn-0.11.0/kernels/flash_api.cu:5-17` (VERIFIED); review panels (A3).

---

### ADR-009: Precision policy

**Status:** Accepted. fp16 and fp8 admission is Deferred-until-measured. **Area:** A3.

**Context.**
- On a 4090 the CUDA reference runs all three shipped checkpoints under bf16 autocast, because every config says `amp_dtype: "bf16"` [CR C1]. That was read in the EN and multilingual configs at `1c5edc17`, and typed-decisions inherits the EN config; M0 re-reads the configs at the `c5d78730` and `f9ab0b22` pins (ADR-005).
- Weights are fp16 on disk (VERIFIED by proxy [CR G1]).
- The multilingual encoder has a +3.3e4 GeGLU activation outlier [EA §6.2].
- cuBLASLt on Ada supports only scalar A/B scales for FP8 [RIS §6.5].

**Decision.**

| Mode | GEMM inputs | Accumulate | Residual / LN / softmax / RoPE | Head, scorer, act head | Use |
|---|---|---|---|---|---|
| `fp32` | f32 | f32 | f32 | f32 | CPU default, the golden reference, T3 |
| `bf16` | bf16 | f32 | f32 | f32 | **GPU default for every Laya checkpoint and every dm2 model** |
| `fp16` | fp16 (the stored weights, exactly) | f32 | f32 | f32 | Opt-in per model only when T10 passes (`arbitro sweep-overflow`). At runtime, a non-finite logit re-runs that batch in bf16 and increments a metric. |
| `fp8` | E4M3 W8A8, encoder linears only (Wqkv, Wo, Wi, Wo_mlp) | f32 | f32 | bf16/f32; embeddings and head never quantised | 1.x only. Needs a per-precision `calibration.json` entry and the P11 gates. For Laya checkpoints it is refused unless the user supplies labelled data to `arbitro calibrate`. Never the default in 1.x. |
| `int8` | — | — | — | — | Research only. Naive INT8 moved probabilities by 0.151 on average [EA §6.3]. |

- `precision = "auto"` selects bf16 on GPU and fp32 on CPU. It **never** selects fp16.
- Only the weights of the bf16 GEMMs (the encoder's Wqkv, Wo, Wi and Wo_mlp; the head's in_proj, out_proj, linear1 and linear2; scorer.1; with their biases) are converted from fp16 to bf16 once at load. This equals autocast's per-call cast, because fp16 → fp32 is exact. The embeddings, `type_emb`, every LayerNorm, scorer.3 and the act head keep their stored fp16 values and are widened to fp32 inside the kernels, as the reference keeps them fp16-exact inside fp32 parameters. The fp16 copy of the GEMM weights is kept only when fp16 mode is enabled.

*FP8 details (M9):*
- Weight scales are per output channel. Activation scales are per token and are computed inside K4/K5; per-token scales preserve batch invariance.
- The GeGLU SmoothQuant trick is exact: divide the rows of Wi that produce the linear half u₂ by sᵢ, and multiply column i of Wo_mlp by sᵢ. This works because g = gelu(u₁)·u₂ is linear in u₂.
- **The planned kernel route is vLLM's CUTLASS 2.x sm89 `scaled_mm` epilogues** (Apache-2.0). The cuBLASLt outer-vector probe is informational only.
- `OpMultiplyAddFastAccum` is checked for accuracy before it is enabled (UNVERIFIED).

**Alternatives.**
- Perf's `auto` → fp16 once the sweep passes: rejected. The reference is bf16, and fp16 is only safe with headroom.
- The RIS rule "fp16 for EN/typed": wrong per [CR C1].

**Consequences.**
- fp16 exists for users who want the exact stored weights.
- The bf16 default carries no overflow risk.

**Validation gate.** T4, T10, P11. `sweep-overflow` reports go to `reports/`.

**Evidence.** [CR C1, G1, G6]; [EA §6.1–6.3]; [RIS §4.6, §6.1, §6.5]; review panels (A3, critical disagreement 5).

---

### ADR-010: Batching and scheduling

**Status:** Accepted. **Area:** A3.

**Context.**
- Batch-1 latency goals are 3–4 ms (P1).
- A 1.5 ms fill wait would nearly halve that budget, and it gains nothing under load, when batches fill anyway.
- TEI's router, queue and backend-thread structure is Apache-2.0 and proven [RIS §3.3].

**Decision.**

*Pipeline:*
1. The tokio frontend parses, validates and normalises.
2. A tokenizer worker pool (physical cores − 2, each worker with its own `Tokenizer`) tokenises the **state once per request**. It is sliced per question with identical ids [EA §8.3]. An LRU keyed by blake3 of the serialized state serves repeated states; another LRU serves header and option ids.
3. A per-model token-budget queue holds work items.
4. One batcher thread per GPU.
5. A backend thread owns the CUDA stream.
6. CPU post-processing workers.
7. The request join.

*Zero-delay batching:*
- An idle GPU launches immediately.
- While it is busy, the next batch is packed up to `max_batch_tokens` (S8).
- `max_wait_us = 0` (S9) is configurable, for throughput experiments.

*Work units:*
- laya-v1: one question sequence.
- dm2: one state group.

Large requests are split across batches at question or option-group boundaries, so interactive traffic interleaves with them.

*Priorities and backpressure:*
- `interactive` and `bulk` classes; per-key token and request buckets.
- Queued items are cancelled when the client disconnects.
- Backpressure starts at `max_queued_tokens` (S10): a request is refused when queued tokens plus its processed tokens exceed S10. Because `max_processed_tokens` is capped at S10 (Q18), an idle server admits every request that passes the limits. ADR-017 defines the status codes.

States over 64 KB may be tokenised in parallel at pre-tokenizer-safe split points only after a fuzz test proves the ids are identical (UNVERIFIED; off by default).

**Alternatives.** Product's `max_wait_us = 1500`: rejected.

**Consequences.**
- Throughput under load comes from queue build-up, not from waiting.
- Batch 1 pays no scheduling tax.

**Validation gate.**
- At concurrency 1, HTTP p50 is within +2 ms of in-process p50 (P1 vs P2).
- Open-loop load tests: no 5xx other than the documented overload codes at 2× saturation.

**Evidence.** [RIS §3.3, §7]; [EA §8.3]; [P-perf A3.4]; review panels (A3, critical disagreement 6).

---

### ADR-011: Determinism and batch invariance

**Status:** Accepted as the default. The *guarantee* is Deferred-until-measured, i.e. UNVERIFIED until the bitwise test passes. **Area:** A3.

**Context.**
- Jev is nondeterministic: noul answers varied 0.46–0.54 over 60 identical calls [BW J5].
- Under dynamic batching, cuBLASLt heuristics can pick a different kernel or split-K for a different M. The last bits then move, and `round4` output can change depending on co-batched traffic [RIS §6.7].

**Decision.**

*`engine.determinism = "batch_invariant"` is the default:*
- one cuBLASLt algorithm per (N, K, dtype), chosen at warm-up across the M buckets and reused for every M;
- split-K = 1;
- FA2 `num_splits = 1`, in both K3 and K3b;
- row-local LayerNorm and element-wise kernels;
- a fixed CPU thread count;
- per-token FP8 scales.

*Claim.* The same request on the same binary, device, precision, model and calibration gives bitwise-identical JSON, whether it runs alone or co-batched with any traffic. Until the test passes, documents must say "designed to be batch-invariant", not "is".

*Test.* Every golden question must give bitwise-identical logits:
- alone;
- inside a 50-question mixed-length batch;
- across token-bucket boundaries;
- under concurrent open-loop load.

*`fast` mode* uses autotuned algorithms. It is opt-in, and it ships only if it is > 5 % faster.

*`x_arbitro.deterministic: true`* makes the server reject the request with a 400 (`api_usage_error`) when the serving backend or mode cannot guarantee determinism.

*If the cost exceeds 10 %* (P10), batch invariance stays the default (Q11), and a fixed-K-chunk CUTLASS GEMM goes on the post-1.0 list.

**Alternatives.** Nondeterministic by default: rejected. Determinism differentiates us from Jev, and the answer cache and paired statistics rely on it.

**Consequences.**
- Up to ~10 % of throughput may be spent (UNVERIFIED).
- The answer cache becomes valid.

**Validation gate.** The bitwise test runs nightly on the self-hosted runner, and P10 is reported.

**Evidence.** [RIS §6.7]; [BW J5]; [P-perf A3.6], [P-acc A3.5], [P-prod A3]; review panels (A3, A11).

---

### ADR-012: laya-compat reference environment and scope

**Status:** Accepted. **Area:** A4.

**Context.**
- Parity needs a pinned reference.
- Laya permits Python ≥ 3.10, but Unicode behaviour differs between CPython versions.
- The router, email cleaning and shortlist cost weeks, and SDK drop-in use does not need them. The heuristic router also misroutes 64 % of short German text [BW F5].

**Decision.**

*Reference environment* (`tools/goldens`, uv lockfile):
- laya 0.3.7 (NandhaKishorM/laya commit `010bacef` = tag `v0.3.7`), installed **from git by commit**: 0.3.7 and 0.3.8 were removed from PyPI by 2026-09-24, when upstream was at 0.3.20, 173 commits past the pin, including changes to `common.py` (`build_sequence` gained `state_ids`, noul gained `labels`) (VERIFIED, AM-17). The parity target stays 0.3.7; moving it is a separate decision with regenerated goldens;
- torch 2.14.0, transformers 5.17.0, tokenizers 0.23.2, numpy 2.4.6;
- **CPython 3.11**, with Unicode 14.0.0 tables. The design phase VERIFIED the reference venv: Python 3.11.15, `unicodedata` 14.0.0.

Parity is *defined* against CPython 3.11, and the documentation says so. Checkpoint revisions follow ADR-005.

*v0.1 scope* (in `arbitro-compat`, on the critical path):
- `pycompat` (ADR-013);
- `render` / `normalise` / `validate`, including the 8 `ValueError` messages in check order;
- `build_sequence`, line by line:
  - the 48-token option cap;
  - `per = max(4, (hml−16)//n)`, counting the marker;
  - `max(8, budget)` for instructions;
  - the mask literal replaced by a space;
  - `st[:room]`;
  - the marker filter and the "options exceed head_max_len" error;
- `temps`: `clamp_temperature` with Python `float()` semantics, bucket precedence, and the exact warning text;
- `post`:
  - float32 softmax with numpy's summation order;
  - entropy confidence;
  - noul max(p, 1−p);
  - f64 expected score;
  - key order and `input_tokens`;
- the act head, whose features use the untempered softmax;
- checkpoint loading (ADR-005);
- `laya` server mode.

v0.1 routing is **`explicit`**: the `model` field or an alias picks the checkpoint.

*v0.2 scope (M4c, 12 dev-h):* `lang` plus the `laya-heuristic` router:
- Python `isalpha` (General Category L*);
- a hand-coded Python `\w`;
- `lower()` length changes;
- `%.0f` half-even;
- tie-breaks;
- Unicode 14 tables.

It passes LIS §13 #30–47 and #63. The residency and thread tests #44–47 are adapted to our registry.

*Community work (self-checking, not scheduled for the maintainer):*
- email cleaning (#54): `fancy-regex`, the 46 literal fixtures, presets copied verbatim;
- shortlist (#48–53): needs an encoder mean-pooling entry point with `add_special_tokens=true` and max_length 512.

The L0 fixtures make these good first issues.

*Quirks reproduced faithfully, and documented:*
- `truncate_left`, including its latent `st[-0:]` bug, behind a flag;
- typed-decisions' inherited EN bucket temperatures [CR C10];
- typed-decisions' train/serve layout skew (512/192 at training vs 1024/256 at serving) [CR G13];
- `encode_special_tokens = false`, so literal special tokens in user text become special ids, exactly as in Python [CR G9].

The docs point users to `arbitro calibrate`.

*Own models* inherit none of these quirks (ADR-019).

**Alternatives.**
- Porting everything into M1 (perf, accuracy): costs weeks before v0.1.
- Dropping Python semantics: silently changes token ids and routing.

**Consequences.** v0.1 users route explicitly. Migrating laya-serve users who rely on auto-routing wait for v0.2.

**Validation gate.** ADR-014, levels L0–L5.

**Evidence.** [LIS header, §1–§11, §13]; [CR C2, C3, C10, C11, G8, G9, G12, G13]; [BW F5]; review panels (A4).

---

### ADR-013: pycompat semantics (Python-compatible serialisation and arithmetic)

**Status:** Accepted. **Area:** A4.

**Context.**
- `build_sequence` tokenises Python-serialised JSON.
- A single escaping or float-formatting difference changes token ids [CR C11].
- Perf's escape rule was wrong for `ensure_ascii=True`. The design phase re-verified the correct behaviour in the reference venv.

**Decision.** `arbitro-compat::pycompat` is a dependency-light module, and we offer it upstream.

*Which `json.dumps` call applies where:*

| Input | Python call | Mode |
|---|---|---|
| state | `json.dumps(state, ensure_ascii=False)` | default separators |
| criteria | `json.dumps(v, ensure_ascii=False, separators=(", ", ": "), default=str)` | Python's defaults, *with* spaces |
| non-string instructions | `json.dumps(ins)`, then `str(...)` | ensure_ascii=True |

*Escape table* (VERIFIED 2026-09-23 in the reference venv: `json.dumps('a\x7fb\u2028c\x1f\u00e9\U0001F600')` gives `"a\u007fb\u2028c\u001f\u00e9\ud83d\ude00"`; with `ensure_ascii=False` the DEL and U+2028 characters are emitted raw, U+001F becomes `\u001f`, and é and the emoji are emitted raw):

| Character | `ensure_ascii=False` | `ensure_ascii=True` |
|---|---|---|
| `"` and `\` | `\"` and `\\` | same |
| `\n \r \t \b \f` | short escapes | same |
| other U+0000–U+001F | `\u00XX` (lower-case hex) | same |
| U+007F DEL | **raw** | `\u007f` |
| U+2028 / U+2029 | **raw** | `\u2028` / `\u2029` |
| other non-ASCII in the BMP | raw | `\uXXXX` (lower-case hex) |
| astral code points | raw | surrogate pair, e.g. `\ud83d\ude00` |

*Numbers:*
- Parsed as raw text (serde_json `arbitrary_precision` + `preserve_order`).
- Integers are exact big integers. The integer `-0` becomes `0`; the float `-0.0` stays `-0.0`.
- Floats go through f64 and are printed with Python `repr`: shortest round-trip digits; fixed notation for −4 ≤ exp < 16, always with ".0"; otherwise `1e+16` / `1.5e-05` style.
- `NaN`, `Infinity` and `-Infinity` are emitted as literals; `1e400` becomes `Infinity`.
- Fractional input numbers are parsed to f64 first, as FastAPI does.

*Objects:* insertion order is kept. A duplicate key keeps its first position and takes the last value.

*Formatting:*
- `%r` repr, with Python's quote-choice rules;
- `%.0f` rounds half-even;
- `round(x, 4)` on the exact binary value, via correctly rounded decimal formatting.

*numpy:* the float32 softmax denominator and the entropy reproduce numpy 2.4.6's pairwise summation order exactly. L0 vectors cover k ∈ {1…255}.

*Unicode tables:* `tools/goldens/gen_pyunicode.py` generates them once from CPython 3.11 `unicodedata`, and they are committed. They cover:
- `isalpha` (GC L*) and `isalnum`;
- Python `re` `\w`;
- full `str.lower` mappings, for example 'İ' → 2 code points.

Rust's `is_alphabetic()` and the `regex` crate's `\w` are never used on the compat path [CR G8].

**Alternatives.** Approximate emulation: rejected, because builder bugs would become indistinguishable from numeric error.

**Consequences.** About 1–2k lines of carefully tested code, reusable outside the project.

**Validation gate.**
- L0 property tests against a Python subprocess: 10k cases per PR, 10⁶ nightly.
- The mutation self-check (ADR-014).

**Evidence.** [CR C11, G8]; [LIS §1.3–1.5, §7.2–7.3, §8.2]; a verification run in the reference venv (design phase); review panels (A4).

---

### ADR-014: Parity harness, fixtures and tolerances

**Status:** Accepted. **Area:** A4.

**Context.**
- Numeric parity must be localisable: a failure should point to one layer.
- The reference differs by backend. The CPU reference takes PyTorch's fused `_transformer_encoder_layer_fwd` path; CUDA runs unfused under autocast [CR G12].
- The upstream MASSIVE table was produced with laya 0.2.0 on torch 2.8, so it contains near-ties. Exact per-language equality is too brittle.

**Decision.**

*Levels:*

| Level | What | Fixtures | Gate | Runs |
|---|---|---|---|---|
| L0 | Pure functions: `pycompat`, render, validate, temps, post, loading (v0.2 adds lang and router; community adds shortlist and email) | LIS §13 as literal vectors: #1–29 in v0.1, #30–47 in v0.2, #48–54 when ported. Plus ≥ 5k generated inputs per function. | Byte-identical (T2) | Every PR (CPU) |
| L1 | Tokenisation + `build_sequence` | ≥ 10k random items per tokenizer: all qtypes; k ∈ {1, 2, 3, 5, 11, 20, 77, 125, 150, 255}; non-ASCII; JSON states with floats and big ints; embedded `[MASK]`/`[SEP]`/`<eos>` literals; over-budget cases | T1 | Every PR |
| L2 | Model math on tiny random-weight models `test-tiny-en` and `test-tiny-multi` (D = 128, 6 layers, window 16, ragged lengths > 129), built with `from_config` and fixed seeds | Committed fp32 safetensors (a few MB of random weights) plus logits and act logits | T6 (CPU); T5 on CUDA (debug fp32 path, Q17) | Every PR (CPU); nightly (CUDA) |
| L3 | Real checkpoints at the pins | A 2k-question golden set: k ∈ {1, 2, 77, 125, 255}, score levels 2–10, noul without criteria, full context, the `choice:11+` bucket, special-token literals, the laya-mlx 16-case set. Only outputs are committed. | T3 (CPU fp32); T4 (GPU bf16); T5 | Nightly on the self-hosted 4090; locally via `cargo xtask parity` |
| L4 | Upstream tables | feishu_zh, 128 requests (#61); the MASSIVE 51-language sweep (#62); the router golden for 11 languages (#63, from v0.2) | T7, T8 | Release candidates |
| L5 | End-to-end HTTP in `laya` mode | The laya-serve cases (#55–60) on the tiny model; replayed feishu requests on real weights | Byte-identical JSON, except the documented deviations: the request-id header is always sent, internal errors return 500 instead of 422, and `/v1/models` exists | Every PR (tiny); release candidates (real) |

*Fixture policy:*
- Small fixtures (ids, logits, answers) are committed. Large ones are versioned release assets with a sha256. Weights are never committed.
- Both `tokenizer.json` files (EN 3.6 MB; multilingual 34 MB, Gemma-derived) are fetched at the pinned revision and verified by sha256 via `cargo xtask fetch-test-assets`, then cached. They are never committed.
- The tiny models are random initialisations. They are not Laya weights.
- A weekly job regenerates the references and diffs them.

*Checks built into the harness:*
- **Mutation self-check.** CI flips known quirks one at a time and asserts that L1 (or L0) fails: `st[:room]` → `st[-room:]`; `per` excluding the marker; ensure_ascii for instructions.
- **Near-tie handling.** T3 exempts, and lists, argmax disagreements where the reference top-2 margin is < 1e-3. T8 allows ±1 item per language, but only on near-ties.
- **Act logits** (about ±4,000 in magnitude [BW §2.3]) are reported separately and are not gated.

**Alternatives.**
- tch/libtorch as an in-CI oracle: a 2–4 GB dependency.
- End-to-end-only parity: does not localise failures.
- Perf's L4 target of 2e-3 max: unrealistic, because the reference's own bf16 quantisation already exceeds it.

**Consequences.**
- CI needs no weights.
- The nightly job needs the self-hosted runner (ADR-029).

**Validation gate.** The table above. v0.1 requires L0–L3 green for all three checkpoints.

**Evidence.** [LIS §13]; [CR G12, G1]; [BW §2.2, §2.3, §4.1]; [RIS §6.4]; [EA §0, §10.7]; `laya/research/results/cpu_51_language_sweep.json` meta (`laya 0.2.0`, `torch 2.8.0`), VERIFIED during the design phase; review panels (A4).

---

### ADR-015: Server modes, routes and wire contract

**Status:** Accepted. **Area:** A5.

**Context.**
- Drop-in users send `model: "jev-latest"`, sometimes send list criteria, and need the request-id header [JAS §9.4].
- Some users want Jev's exact contract, for testing.
- laya-serve users want byte compatibility.
- Mode names must carry no third-party marks.

**Decision.**

*Routes:* §3.4.
- `/ready` stays false until warm-up and graph capture are done.
- The path aliases are off by default (`server.path_aliases`).

*Modes* (`server.mode`). A mode changes validation and output numerics, never the model.

| Behaviour | `strict` | `lenient` (default) | `laya` |
|---|---|---|---|
| Purpose | Test clients against Jev's contract | SDK drop-in | Migrate laya-serve users |
| `model` missing | 422 `missing` | → `models.default` | Routing (`explicit` → default) |
| Values in `models.aliases` (`jev-latest`, `jev-preview`) | Accepted → default | Accepted → default | Routing (laya-serve sends `jev-*` to auto-routing, LIS §13 #55, #60) |
| Other unknown ids (e.g. `jev-1.13.0`) | 400 `{"detail":{"error_type":"api_usage_error","message":"Unknown model: X"}}` | Other `jev-*` → default; any other id → the same 400 | Routing |
| List-form choice criteria | 422 | Accepted (extension) | Accepted |
| `type: "boolean"` | 422 | Treated as noul | Laya's behaviour |
| `state` null or missing | 422 | 422 | Serialised as `"null"` [LIS §1.5] |
| Empty `questions` | 422 | 422 | 200 with an empty answer set [LIS §1.3] |
| > 255 options / > 10 levels | The exact Jev 400 bodies | same | No cap beyond the model budget |
| Option overflow of the model budget | 400 `max_tokens_exceeded` with a message | same | Laya's error string |
| Validation rules | The OpenAPI schema | The OpenAPI schema plus the extensions above | Laya's `_check_question` [LIS §1.2] |
| Missing API key | 403 | 401 | 401 |
| Choice confidence | Rescaled peak | Rescaled peak | Entropy confidence |
| Score confidence | `output.score_confidence` (default `peak`) | same | Entropy confidence |
| Noul fields | `{type, noul}` | `{type, noul}` | Adds `confidence` and `action` |
| Default rounding | `round2` | `full` | `round4` |
| Extras | None unless requested via `x_arbitro` | Under `x_arbitro`, when requested | `routing`, `action.act_probability`, `model: "laya-rl-agent"` |

*Aliases* are never listed in `/v1/models` unless `models.list_aliases = true`. Responses carry the concrete id, except for `laya` mode's `model: "laya-rl-agent"` (row "Extras"); the `x-arbitro-model` header always carries it.

*Conformance suite* (`tests/conformance`, every PR, against a server backed by `test-tiny-en`). Clients, all unmodified and pinned by Renovate so that SDK drift shows up as a red test:
- `typesafe-sdk` 0.7.1 (Python);
- `@typesafe-ai/sdk` 0.6.0;
- `@ai-sdk/typesafe-ai` 3.0.4;
- `pydantic-ai-slim` 2.48.0;
- optionally, nightly: `hs-jev`.

It covers:
- every qtype;
- every error status and body;
- the request id;
- retries on 429/503/529 with `retry-after-ms`;
- `models.list()`;
- tolerance of extra fields;
- the third-party validator checks: `choice ∈ criteria`, `set(probabilities) == set(criteria)`, \|Σ − 1\| < 0.02, and `probabilities[choice] ≥ max − 1e-6`.

Recorded third-party Jev responses are used **only** as response *shapes*. They are referenced by URL and commit, never vendored (ADR-031).

**Alternatives.**
- Product's `jev` / `jev-strict` mode names: rejected on trademark hygiene.
- Strict by default: breaks SDK defaults.
- Per-request mode switching: produces ambiguous support tickets.

**Consequences.** Three modes to test; the conformance suite covers all three.

**Validation gate.** Conformance 100 % on every PR. The L5 laya-mode replay.

**Evidence.** [JAS §3.1–3.7, §9.1–9.4, §10, §12 #11]; [LIS §1.2–1.5, §11]; [P-prod A5], [P-perf A5], [P-acc A5]; review panels (A5, A9).

---

### ADR-016: Output numerics and confidence statistics

**Status:** Accepted. **Area:** A5.

**Context.**
- Jev rounds to 2 decimals, produces hard zeros and sums of 0.99, which cause NLL blow-ups: gold p = 0 on 15–16 % of emotion items [BW J3].
- JAS §9.4 #6 says the server SHOULD emit full precision summing to 1 within 1e-6, with no zeros.
- Product's 4-decimal default with a 1e-4 floor moves about 2.5 % of the probability mass at k = 255 and caps NLL resolution.
- Choice confidence is (n·peak−1)/(n−1) [CR C4]. Score confidence fits peak best (81/150 samples vs 48/150 for the formula), but that evidence comes from one file [CR G7].

**Decision.**

*Rounding modes:*
- **`full`** (the `lenient` default):
  - probabilities are computed in f64 from the f32 logits and the calibration;
  - each pᵢ becomes max(pᵢ, 1e-6), then the distribution is renormalised;
  - the sum equals 1 within 1e-9, and there are no hard zeros;
  - values are serialised as shortest round-trip f64.
- **`round2`** (the `strict` default) approximates the observed Jev output format. Jev's exact rounding rule is unknown (INFERRED; see [ANALYSIS.md §5.3](ANALYSIS.md)): per-entry flooring plus one +0.01 correction [CR G7] fits the priorbench and Feishu logs, but the DMB direct-API log has float artefacts on more than one entry in 52 of 3,900 answers, which fits cumulative-sum differencing [JAS §3.4] better. `round2` therefore guarantees only the properties seen in every log:
  - 2-decimal values, sum ≤ 1 and never above (0.99 allowed);
  - noul is clipped to [0.01, 0.99];
  - `choice` is the argmax of the *unrounded* calibrated probabilities (one DMB answer picks a label shown at 0.14 over one shown at 0.15), with ties broken by criteria order;
  - confidence is computed from the *unrounded* calibrated p, then rounded.
- **`round4`** (the `laya` default): Python `round(x, 4)` semantics via `pycompat`.

A request can override the mode with `x_arbitro.rounding`.

*Confidence:*
- **Choice** (`strict`, `lenient`): rescaled peak (n·p_max − 1)/(n − 1), clipped to [0, 1], 1.0 when n = 1, computed on the unrounded calibrated p.
- **Score:** `output.score_confidence = "peak"` by default. `rescaled_peak` is available. This is a documented, deliberate choice: the [CR G7] empirical fit overrides the documented formula in [JAS §9.4 #7], and the evidence is thin.
- **`laya` mode:** entropy confidence 1 − H/ln k [LIS §7.3].

*Shapes:*
- Choice `probabilities` follow criteria order, not Jev's random order.
- `answers` follow request order.
- Score: `score` = Σ i·pᵢ (f64); `legend` echoes the criteria under the keys "0"…"n−1"; `probabilities` uses the same keys.
- Noul has no `confidence` field in `strict` / `lenient`.

**Alternatives.**
- Product's `decimals = 4`, `prob_floor = 1e-4`: rejected. Of the two review panels, one accepted it and one rejected it; JAS §9.4 #6 and the "optimal accuracy" goal decide the tie.
- The formula for score confidence in `lenient`: available as a config value.

**Consequences.**
- Clients that string-compare Jev's two-decimal output must opt into `round2`.
- NLL-based evaluation of our outputs is never distorted by rounding.

**Validation gate.**
- Property tests: sum, floor, no zeros, argmax consistency, legend key shapes.
- `round2` output reproduces the shape statistics of third-party logs: sums in {0.99, 1.00}, never > 1.
- The conformance validator checks (ADR-015).

**Evidence.** [JAS §3.4, §9.4 #6–#7, #10]; [CR C4, G7]; [BW J3]; [LIS §7.3]; review panels (A5, critical disagreement 4).

---

### ADR-017: Errors, deadlines, overload, limits and auth

**Status:** Accepted. **Area:** A5.

**Context.**
- The SDK per-attempt timeout is 10 s. A slower answer is retried and billed twice [JAS §3.7, §9.4 #19].
- Jev returns 429 for per-account rate limits and 529 for overload [JAS §3.6].
- laya-v1 re-encodes the state for every question, so a large multi-question request can exceed any deadline.

**Decision.**

*Errors:*

| Condition | Status and body |
|---|---|
| Malformed JSON | 422 pydantic `json_invalid` (`strict`, `lenient`). `laya` mode follows laya-serve. |
| Schema errors | 422 `{"detail":[{type, loc:["body",…], msg, input, ctx}]}`, with the discriminator tag in `loc` |
| > 255 options | 400 `{"detail":"Too many choices. Must have at most 255 choices."}` |
| > 10 score levels | 400 with a string `detail`. The exact Jev text is UNVERIFIED [JAS §13 #4–5]. |
| Token limits (S3, S4, `max_processed_tokens`) | 400 `{"detail":{"error_type":"max_tokens_exceeded","message":…}}`. The message names the qid, k and budget, and suggests fewer questions or options. |
| Unknown model | 400 `api_usage_error` (ADR-015) |
| `x_arbitro.deterministic` cannot be satisfied | 400 `api_usage_error` |
| Missing / invalid key | 401 (403 in `strict`) / 401 `{"detail":{"error_type":"authentication_error","message":…}}` |
| Unknown path / oversized body | 404 `{"detail":"Not Found"}` / 413 |
| Per-key rate limit | **429**, with `retry-after-ms` and `retry-after` |
| Global overload (queue above S10) or deadline exceeded | **503**, or **529** in `strict`, with `retry-after-ms` (a drain estimate) and `retry-after` |
| Internal or CUDA fault | 500 `{"detail":{"error_type":"internal_error","message":…}}` with the request id. **Never 422.** A sticky CUDA error sets `/health` to `degraded` and exits the process so a supervisor restarts it. |

*Deadline:*
- `server.request_timeout_ms = 8000` (S5), below the SDK's 10 s.
- `x_arbitro.deadline_ms` can only lower it.

*Limits:*
- S1–S4.
- `limits.max_processed_tokens = "auto"` = the power-of-two floor of 0.5 × deadline × the tok/s measured at warm-up, with a minimum of 4,096 and a maximum of `max_queued_tokens` (S10, Q18). This guarantees P12.
- The limits are counted with the serving model's tokenizer. Jev's tokenizer differs (REPORTED), so limits match Jev's semantics, not its exact counts.
- Per-model `max_len` truncation is always reported (ADR-018), never silent.

*Auth:*
- Bearer keys from `ARBITRO_API_KEYS` or a file of SHA-256 hashes, compared in constant time (`subtle`).
- Keys can be labelled, and each key has token and request buckets.
- Default bind is `127.0.0.1:8080`.
- The Docker images set `ARBITRO__SERVER__BIND=0.0.0.0:8080` and `ARBITRO_HOME=/cache`. Otherwise `docker run -p 8080:8080 -v …:/cache` (the v0.1 definition of done, item 1) could not reach a loopback bind, and the volume would not hold the cache. Inside the container, the non-loopback WARN below applies whenever no keys are set.
- A non-loopback bind without keys logs a WARN every 10 minutes, and `/health` reports `"auth":"disabled"`.

*HTTP:* h1 and h2c; at most 512 concurrent requests (S7); bodies up to 8 MiB (S6).

**Alternatives.**
- Perf's 10 s deadline → 504: equals the SDK timeout, so the client retries at the moment the server gives up.
- Product's 429 for a full queue: overloads the rate-limit semantics.
- Perf's 400 for malformed JSON in lenient mode: diverges from FastAPI.

**Consequences.**
- CPU deployments get small `auto` limits. The docs say so, and give the measured number.

**Validation gate.**
- Conformance error-path tests.
- A load test at 2× saturation returns only 503/529 with retry hints.
- A test that the largest accepted request finishes in < 8 s (P12).

**Evidence.** [JAS §0, §3.6–3.7, §9.4 #15–#19, §10 #16–#17]; [BW §1.3]; review panels (A5, critical disagreement 9).

---

### ADR-018: Extensions, headers, usage accounting, observability

**Status:** Accepted. **Area:** A5.

**Context.**
- The SDKs ignore unknown fields [JAS §9.4 #18].
- The Python answer models are `strict=True` about types. Keeping extras outside the answer objects is the safest placement.
- Silent truncation is a Laya failure mode [BW F10].

**Decision.**

*`x_arbitro` request fields:*

| Field | Meaning |
|---|---|
| `include` | A list drawn from `p_top`, `margin`, `entropy_confidence`, `p_correct`, `p_none`, `decision`, `truncation`, `logits`, `act`, `timing`, `routing` |
| `rounding` | `full` \| `round2` \| `round4` |
| `permutations` | K-order averaging, opt-in, costs K× compute |
| `automate_alpha` | Conformal error budget |
| `priority` | `interactive` \| `bulk` |
| `deterministic` | bool; the request is rejected if determinism cannot be guaranteed |
| `deadline_ms` | Can only lower the server deadline |
| `shortlist_k` | Only once shortlist exists |

*`x_arbitro` response fields:*
- Top level: `ext_version`, `backend`, `precision`, `calibration_id`, `model_family`, `timing_ms{tokenize,queue,forward,post}`, `routing`.
- Per answer, placed at `x_arbitro.answers.<qid>`:
  - `p_top`, `margin`, `entropy_confidence`;
  - `p_correct`, `p_none`, `decision`;
  - `truncated_state_tokens`, `truncated_option_tokens`, `options_kept`;
  - `logits`, `act_probability`.

*Headers.* Always sent:
- `x-typesafe-request-id: req_<32 hex>` (wire-required);
- `x-arbitro-model`;
- `x-arbitro-truncated-questions: <n>`;
- `server-timing: tok;dur=…, queue;dur=…, gpu;dur=…, post;dur=…`.

`x-request-id` is echoed when the client sends one. With CORS enabled, `Access-Control-Expose-Headers` includes the request-id header.

*Usage*, in every mode:
- `input_tokens` is the number of **tokens actually processed**. For laya-v1 that is the sum over the per-question sequences, which equals laya-serve's number. For dm2 layout L2 it is the state once plus the question suffixes.
- `output_tokens` = 0.
- Documented as "honest compute, not Jev billing" [JAS §9.4 #11].

*Observability:*
- `tracing` JSON logs: request id, model, number of questions, tokens, queue wait, forward time, status.
- **Bodies are never logged** unless `log_bodies = true`.
- Prometheus metrics prefixed `arbitro_`:
  - `requests_total{model,status}`, `request_seconds`;
  - `queue_tokens`, `batch_tokens`;
  - `forward_seconds{backend,bucket}`, `tokenize_seconds`;
  - `truncated_questions_total`;
  - `model_resident{model}`, `graph_hits_total`;
  - `nonfinite_fallbacks_total`;
  - NVML memory, clock and power.
- OTLP behind a feature.
- **No telemetry** is ever sent anywhere.

**Alternatives.**
- Extras inside each answer object (perf, accuracy): rejected as an avoidable SDK risk.
- Counting the state once in strict mode for laya-v1 models (accuracy): rejected, because it under-reports compute [JAS §9.4 #11].

**Consequences.** `x_arbitro` has a version (`ext_version`), which is frozen at 1.0.

**Validation gate.**
- Conformance tests check that extras never break any SDK.
- Unit tests check `usage` against hand counts.

**Evidence.** [JAS §9.4 #2, #11, #18]; [BW F10]; [P-prod A5], [P-perf A5]; review panels (A5).

---

### ADR-019: dm2 architecture and the layout programme

**Status:** Accepted for the fixed parts. Layout, read-out and head are Deferred-until-measured: the M5 ablations settle them, and their results are frozen in `docs/adr/ADR-019a-dm2-frozen.md`. **Area:** A6.

**Context.**

Laya's weaknesses are in the model, not the runtime [BW §0]:
- near-chance zero-shot accuracy;
- a noul answer that follows its label tokens;
- 15–23 % order flips;
- a 192/256-token shared option budget that caps options at about 125–250;
- a truncated state;
- a saturated act head.

Laya also re-encodes the state once per question (N×), whereas Jev's latency is flat in N [JAS §4].

**Decision.**

*Fixed parts* (not ablated):
- **Marker.** Each option gets the pretrained `[MASK]` token as its marker. Fresh marker tokens do not learn [LTR §3; kotoba].
- **Engine target.** A ModernBERT-architecture encoder, so the FA2 engine applies.
- **Hardened tokenisation.** `encode_special_tokens = true` [CR G9]. That flag only covers added tokens flagged `special`, and the `[unused*]` / `<unused*>` tokens used as question-type tokens are not (VERIFIED on the shipped `tokenizer.json` files, TRAINING.md §4.4). The builder therefore inserts type tokens by id, and the dm2 bundle's tokenizer marks them special.
- **Per-option spans.** `[MASK]` plus up to 32 text tokens (the cap is ablated in X6).
- **Option-group chunking.** Groups of ≤ 64 options. Each group is its own suffix over the same state, and all groups' logits go through one joint softmax. 255 options × 33 tokens = 8,415 tokens would overflow any single budget, so 255 options become 4 groups of ≤ 2.2k tokens each.
- **Chunk invariance.** Training randomly chunks 20 % of choice questions. Gate: chunk invariance ≤ 0.02 (G-Q6).
- **Explicit none option.** A virtual last option, "none of the options applies". It is always present at inference and present in 50 % of training questions.
  - `p_none` is its mass.
  - Jev-compatible `probabilities` are renormalised over the real options only.
  - X5 gates the cost on answerable items at ≤ 0.5 pp.
- **Dual channel.** See ADR-021. The act head is dropped.
- **Context.** 8,192 tokens, native RoPE. 10–20 % of training steps use 2–8k-token states. Beyond 8k, v0.3 truncates head+tail and reports it. BM25/salience packing is post-1.0 research.
- **Question types.**
  - noul: two neutral options, with random order, label-swap augmentation (the true/false descriptions swap together with the target) and negation pairs.
  - score: levels as options, rendered as "level i: …", with RPS in the loss and random scale reversal.
  - choice: options shuffled in every epoch [CR C8].
- **No train/serve skew.** Training uses the serving layout, through the Rust data core (ADR-023).

*Layouts ablated in M5 (X2).* An early signal on ModernBERT-base comes from E1.

| Layout | Attention | State cost | Role |
|---|---|---|---|
| L0 | One sequence per question, question then state, fully bidirectional | N × state | Baseline and final fallback |
| **L2** (prefix-isolated) | `[CLS] state [SEP]` attends only to itself. Each question suffix `<type> instructions [SEP] [MASK] opt … [MASK] none [SEP]` attends to state ∪ itself. RoPE positions continue after the state. | 1 × state | Preferred if it passes the rule |
| L3-k | L2 for the lower layers; the top k ∈ {2, 4, 7} layers re-encode `[state; question]` per question | (N_L−k)/N_L + k·N/N_L | Fallback if L2 fails |
| T (arm) | Isolated option branches with tied positions, plus a 2-layer set-transformer head without positional encoding | 1 × state | Ablation only. It needs the "level i:" rendering for score questions. FA2 wastes about 5× of the QK/PV tile work on short branches in global layers, so it is not competitive without a multi-range kernel, which is not planned. |
| L1 | All questions in one sequence | — | **Rejected.** Answers would depend on co-asked questions [JAS §3.3]. |

*Pre-registered decision rule* (written into `docs/ablations.md` before any run):
- Take the cheapest layout whose OOD-S *dev* macro accuracy is within 1 pp of layout L0, with a paired-bootstrap CI lower bound > −1.5 pp.
- Otherwise take the smallest passing k of layout L3.
- Otherwise ship layout L0 in v0.3 and move shared state to a later release.
- Layout T is adopted only if it passes the same rule vs layout L0 **and** its measured serving cost on P9 is ≤ that of layout L2.

*Read-out and head (X3):*
- Arms: marker-only / span-mean / hybrid `z_j = MLP(LN([h_mask_j; mean(h_opt_j); h_q ⊙ h_mask_j]))`, each with `head_layers` ∈ {0, 2}.
- Prefer `head_layers = 0` when it is within 0.5 pp; that is less engine code, and the head is 7–12 % of FLOPs [EA §7.2].
- If a head is kept, it gets a final LayerNorm, because its absence causes Laya's ~1e4 residual norm [BW §2.3].

*Other ablations:*
- X4 loss: `w_sph` ∈ {0, 0.25, 0.75}; perm-KL ∈ {0, 0.1, 0.5} on top of shuffling [CR C8]; teacher α ∈ {1.0, 0.8, 0.6}.
- X5 none and unknowable share.
- X6 chunking and span cap.
- X7 rendering of JSON states (JSON vs labelled lines vs mixed).

*Known limitation of layout L2.* In windowed layers a question token sees only the last 64 state positions. Full state access comes from the global layers (every third). Layout L3 partly exists to test whether this matters.

*Engine for layout L2* (M6, only if layout L2 is adopted):
- K3b paged split-KV FA2 with `num_splits = 1`, page size a multiple of 32, and copy-on-write of the last partial state page per question.
- A state K/V cache across requests (MEM5: 112 KiB/token large, 66 KiB/token base).
- Last-layer skip for state rows: only their K/V is needed, so the attention output, Wo and the MLP are skipped.
- The gather-then-dense path is the reference implementation, and it is the path candle uses.

*Training for layouts L2 and L3:* gather the state K/V per question and call `varlen_attn` with `cu_seqlens_q ≠ cu_seqlens_k`; autograd handles the fan-in. We never rely on a paged varlen backward [CR G4, G5].

*v0.3 scope (must-have):*
- marker (or hybrid) read-out;
- the adopted layout;
- chunking;
- the none option;
- the dual channel with conformal thresholds;
- augmentation;
- 8k context with head+tail truncation.

*Deferred past v0.3:*
- salience/BM25 packing;
- APS/RAPS prediction sets;
- a cumulative-link ordinal head;
- multilingual.

**Alternatives.**
- Perf's tree as the default: it stacks several unproven changes, removes cross-option attention, and roughly doubles the paged-attention engine work.
- Product's 32-token spans under a 4,096-token cap: cannot hold 255 options.
- Fresh marker tokens: they do not learn.

**Consequences.** The dm2 engine path depends on X2. Laya-style layout L0 stays fully supported, so a failed shared-state bet costs speed, not a release.

**Validation gate.**
- The ablation reports `reports/ablation-X*.json`, with CIs.
- T9 (paged-KV ≡ gather-then-dense ≤ 1e-5 fp32, on K13's block-table mode, Q17).
- P9.
- G-Q1…G-Q6 (ADR-022).

**Evidence.** [CR G4, G5, G9, G10, G11, C8]; [LTR §3, §10]; [BW F7–F10, J3–J4, §2.3]; [JAS §3.3, §4]; [EA §7.2]; kotoba README; verdict2 [CR G10]; review panels (A6, critical disagreements 2 and 14).

---

### ADR-020: Backbone selection and the distillation rule

**Status:** Accepted as a procedure. The choice itself is Deferred-until-measured: E1 gives an early signal by about week 11, and X1 decides in M5. **Area:** A6.

**Context.**

Evidence against ModernBERT-large cold start:
- kotoba: ModernBERT-large stayed at the label prior (0.388–0.399) at 3k states / 1 epoch across every learning rate tried, with fresh-marker and span heads. DeBERTa-v3-large reached 0.787 on the same data, and 0.855 at 18k states (Q-ref11).

Evidence for ModernBERT-architecture models:
- Laya's multilingual head was trained **from scratch** on mmBERT-base, a ModernBERT-architecture model, and learned: 15,987 updates, 4 epochs (Q-ref12).
- verdict2's ModernBERT-base reaches 0.771 (Q-ref3).
- Laya EN is a working ModernBERT-large. It was fine-tuned from an earlier decision checkpoint, though, so it says nothing about cold start [CR C9].
- kotoba's large runs were short: 1 epoch, 3k states.

Why the choice matters for the engine:
- DeBERTa has no FA2 path and a 512-token context. Serving it would need ORT/TensorRT, a second engine.

**Decision.**

*Risk rating:* likelihood M-H, impact H (ADR-033 R2).

*E1 early signal* (weeks 5–11, GPU nights, 18 dev-h, 10–15 GPU-h):
- Data: a gold-only mini-mixture of ≈ 100–200k decisions from pool-T candidates whose manifests are written during E1 (CLINC150, PAWS, HellaSwag, GoEmotions). No teachers are needed.
- Evaluation: a provisional OOD-S dev set (MASSIVE-en validation split, ANLI dev).
- Setup: layout L0, hybrid read-out, warmup 6 %, LLRD 0.9.
- Arms, 1 seed each at ~100M tokens (plus a second seed for ModernBERT-large):
  - ModernBERT-large;
  - ModernBERT-base;
  - Ettin-encoder-400m, only if its weight licence checks out. Its code repo is MIT (VERIFIED during the design phase); the weight licence is UNVERIFIED because Hugging Face was unreachable from the design environment; it is REPORTED to use the ModernBERT architecture and tokenizer;
  - DeBERTa-v3-large (padded to 512), as the reference arm.
- Also: layout L0 vs layout L2 on ModernBERT-base.
- Output: learning curves on OOD-S dev, and a signal recorded in ADR-020a.

*X1 final* (M5): the same arms, 2 seeds each, on mixture v1 with teacher labels.

*Decision rule* (pre-registered):
- Prefer the best ModernBERT-architecture arm (ModernBERT-large/base, Ettin).
- DeBERTa-v3-large "wins" only if it leads by > 3 pp with a CI lower bound > +1 pp.
- If it wins, it becomes the **teacher / accuracy reference**, and we **distil it into the best ModernBERT-architecture student** for serving on the FA2 engine.
- No second serving engine before 1.0. If the distilled student still trails DeBERTa by > 3 pp, the student ships, and serving DeBERTa via ORT goes on the post-1.0 list.
- `arbitro-en-base` becomes the default if it is within 1 pp of large.

*Perf's condition "the shared-state tree must be expressible in the backbone" is dropped.* It biases the choice toward speed over accuracy.

*Multilingual* (M8, after Q8):
- mmBERT-base (MIT, REPORTED) with frozen or low-LR embeddings.
- It ships only after counsel clears its Gemma-2-derived tokenizer.
- The alternative, `gte-multilingual-mlm-base` (Apache-2.0, REPORTED), has a different architecture and needs engine work (UNVERIFIED).

**Alternatives.**
- Adopting DeBERTa directly for serving: forks the engine.
- Skipping the early signal: the risk would surface in month 8.

**Consequences.**
- About 12–15 GPU-h are spent early.
- A distillation round adds ~30 GPU-h if needed (cut rule C-4).

**Validation gate.** The E1 report by week ~11; the X1 report in M5; both have CIs.

**Evidence.** [CR C9, G10, G14]; [LTR §0 table, §3]; kotoba README:97–187; [RT §4.6]; Ettin repo LICENSE (design phase); review panels (A6, A11, critical disagreement 3).

---

### ADR-021: Dual-channel output and calibration

**Status:** Accepted. **Area:** A6.

**Context.**
- Soft labels make a good distribution fit (Brier) and top-1 correctness calibration (ECE) conflict. verdict2 resolved this with a separate correctness head (ECE 0.0144) [CR G10].
- Laya's temperatures were fitted on training items. Its `choice:11+` value of 0.1006 is degenerate, and its act head has AUROC 0.30 [LTR §7; BW F1, F13].
- The temperatures a model needs vary from 1.5 to 8.8 across languages [BW §1.4].

**Decision.**

*Channel 1: calibrated distribution.*
- Hierarchical temperatures T(qtype, k-bucket) with shrinkage.
- A feature-conditioned log T (script/language, log state length, truncation).
- Platt scaling for noul.
- Fitted **only** on data held out by group (group = state id).
- Shipped calibration artefacts (both channels and the conformal thresholds) are fitted only on items whose licence allows training. NC-licensed pool-O suites (e.g. ANLI, toxic-chat) are used for evaluation and model selection only, until counsel says otherwise (Q13) [TRAINING.md §10.4].
- `calibration.json` is keyed by precision and records the hash of the fitting split.
- Any temperature at a clamp bound fails the gate.
- `confidence` = rescaled peak on this distribution (ADR-016).

*Channel 2: `p_correct`.*
- A logistic model or 2-layer MLP over: p_max, margin, entropy, log k, qtype, script id, log state length, truncation flags, p_none, and the K-permutation disagreement when `permutations` is on.
- It is fitted **out-of-fold** on held-out predictions. It is never trained jointly with the model and never on training items.

*Conformal gating.*
- Learn-then-Test thresholds on `p_correct` for α ∈ {0.01, 0.02, 0.05, 0.10}, δ = 0.05.
- The result is `decision` = `automate` | `review` under `x_arbitro.automate_alpha`.
- APS/RAPS prediction sets come after 1.0.

*Laya checkpoints.* `arbitro calibrate` refits temperatures on the user's labelled JSONL, using a held-out split, and writes a user `calibration.json`. This is laya-plus and is reported separately.

*Documentation* recommends gating automation on `p_correct` / `decision`, not on `confidence`.

**Alternatives.**
- Temperature-only calibration: does not solve the Brier-vs-ECE conflict.
- A jointly trained act head: that is Laya's failure mode.

**Consequences.** There is a post-training fitting stage (in Rust, `arbitro-eval::calib_fit`), and it must run for every precision.

**Validation gate.**
- G-Q3 and G-Q4.
- Calibration is always reported three ways: raw (T = 1), shipped, and held-out refit (ADR-027).

**Evidence.** [CR G10, C10]; [LTR §7, §10.4–10.5]; [BW F1, F13, §1.4]; [P-acc A6.2(i)].

---

### ADR-022: dm2 release gates

**Status:** Accepted. **Area:** A6.

**Context.**
- Product's proposal had no quantified bar for beating Laya.
- Accuracy's G-Q1 had no fallback.
- Some suites double as Jev comparisons, so reading their test splits repeatedly must be disclosed.

**Decision.**

*Gates.* All are paired comparisons on the same items with record-clustered bootstrap 95 % CIs. "Laya" means `laya-en` run by us through the compat runtime. Model claims use 3 seeds (mean ± sd).

| Gate | Criterion | If it fails |
|---|---|---|
| **G-Q1 (primary, pre-registered)** | OOD-S test macro accuracy Δ ≥ +10 pp vs `laya-en`, CI lower bound > +5 pp. NLL and Brier better, CIs excluding 0. | CI lower bound > 0 but < +5 pp: ship as `arbitro-en-large-<ver>-preview`, with honest numbers and no "beats Laya clearly" claim. CI lower bound ≤ 0: no release; return to M5. |
| G-Q2 (reported, not blocking) | Recover ≥ 50 % of the (Jev − Laya) accuracy gap, macro-averaged over pool-J suites; third-party published Jev numbers only | Reported honestly as not reached |
| G-Q3 | Distribution ECE-15 ≤ 0.05 (in-domain fit, evaluated on OOD-S test); `p_correct` ECE ≤ 0.03; no temperature at a bound | Blocks |
| G-Q4 | Coverage at ≤ 5 % error ≥ 0.60; unknowable items answered at ≥ 0.9 confidence ≤ 5 % | Blocks |
| G-Q5 | Flip rate ≤ 0.05 on the 20-option MASSIVE-en permutation probe; noul label-swap consistency ≥ 0.90; shuffled-context accuracy within 2 pp of the label prior; question isolation bitwise (layout L2) | Blocks |
| G-Q6 | Code-word test (DMB S3 protocol) ≥ 0.98 at k = 255; Banking77-77 (a held-out source) ≥ `laya-en` + 20 pp (goal ≥ 0.75; Jev published 0.763); chunk invariance ≤ 0.02 | Blocks |

*Hygiene gates:*
- licence gate passed;
- MinHash/13-gram overlap of pool T vs pools O ∪ J ∪ L is below threshold, and every flagged item is removed;
- group splits hold;
- T9 export parity;
- P9 when layout L2 is adopted.

*Test reads.*
- Each test split is read once per release.
- Every read is appended to `reports/test-reads.jsonl`.
- If a gate fails and we iterate, the next report states the read count.

*Evidence notes the model card must carry:*
- Kev's second-pass gain (0.837 → 0.852) came from uniform-target unknowables **combined with** policy cases that use explicit day counts (kev README:137, [RT §6.2]). We cite it as evidence for both, not for unknowables alone.
- JevBench results are tagged `held-out-source` (the family is seen), because our synthetic families overlap its categories by construction (ADR-025).

**Alternatives.** "Non-inferiority" only (perf): too weak for a product whose raison d'être is accuracy.

**Consequences.** A release can slip if G-Q1 fails. The preview path still ships something honest.

**Validation gate.** The table itself, evaluated by `arbitro eval --release` and checked against `reports/claims.toml`.

**Evidence.** [BW §0, §1.1–1.4, F8]; [RT §8.5]; [CR G10]; [P-acc A6.1]; kev README; review panels (A6, A8).

---

### ADR-023: Training stack

**Status:** Accepted. The P1 part is Proposed-needs-user-input (Q3). **Area:** A7.

**Context.**
- No Rust framework has varlen/windowed flash backward plus AMP [RT §0, §3]:
  - burn has no flash backward and no AMP;
  - candle's fused ops cut gradients;
  - tch has fp16-only autocast and no varlen autograd.
- PyTorch 2.14's `varlen_attn` has a window and a backward on sm_89 [CR G4].
- Train/serve skew hurt Laya [CR G13].

**Decision.**

*Trainer* (`training/`, PyTorch 2.14):
- Our own ~300-line packed ModernBERT module, built on `torch.nn.attention.varlen.varlen_attn`, with window (64, 64) on local layers and (−1, −1) on global layers.
- bf16 autocast with **fp32 master weights**. Without them, 82–91 % of each update is lost at LR 1–3e-5 [RT §4.4].
- Fused AdamW.
- `torch.compile` over a **static** 12,288-token micro-batch, padded with a dummy tail segment; variable shapes were 6× slower.
- No activation checkpointing at ≤ 12k tokens. Gradient accumulation × 4 ≈ 49k tokens per step (C3).
- The scorer's last projection in fp32.
- The mmBERT embedding frozen.

*Optimiser:*
- AdamW β = (0.9, 0.98), eps 1e-6, weight decay 0.01, excluding norms, biases and embeddings;
- LR 2e-5 for the encoder and 3e-4 for the new head;
- LLRD 0.93 (0.9 / 0.95 are ablated);
- warmup 6 %, cosine schedule, clip 1.0, EMA 0.999.

*Loss:* `L = soft-CE (log floor −9.21) + w_sph·(−spherical) + w_rps·RPS·[score] (+ λ_pkl·KL_sym only if X4 selects it)`, starting with w_sph = 0.5 and w_rps = 1.0.
- **No label smoothing**: in Kev's runs, coverage at 5 % error fell from 0.576 to 0.006 with it.
- **Open before M3b:** the hard floor `max(log q̂, −9.21)` has zero gradient wherever q̂ < 1e-4, so confidently wrong items barely train (VERIFIED with torch 2.14, TRAINING.md §9.1). Before M3b this ADR is amended to either a straight-through floor or no floor in the loss; the floor stays in the reported NLL.
- The RLCD evolution-strategies estimator exists only as an `rlcd_es` flag for parity experiments [LTR §4.5].

*The Rust data core, via PyO3 (`arbitro.data`):*
- tokenizer, layouts and augmentation;
- a packer that emits `input_ids`, `position_ids`, `cu_seqlens_q/k`, markers, spans, soft targets, qtype and group ids, zero-copy to numpy;
- a ChaCha RNG keyed by (seed, epoch, example_id), so the number of workers cannot change results;
- training at the serving layout (fixes [CR G13]).

The PyTorch trainer remains the gradient-parity reference for any post-1.0 Rust trainer [RT §7 P5].

*Trainer validation* (the default; licence-clean):
1. Our packed trainer vs a naive padded HF-transformers reference trainer, on the same permissive pool-T subset with the same seeds. Final OOD-S dev accuracy must be within 1 pp, and loss curves within seed noise.
2. A gradient check on the tiny model: packed vs padded gradients within 1e-4.
3. Export parity T9.

*Optional P1* (only if the maintainer says yes to Q3):
- a private, never-published reproduction of Laya's typed-decisions fine-tune;
- it starts from user-downloaded Laya weights;
- target: within noise of the published numbers.

*CI:*
- CPU CI trains the tiny model for 20 steps: the loss must decrease, the export must work, and T6 must hold.
- A nightly 10-minute GPU smoke run.

**Alternatives.**
- A Rust trainer: post-1.0 showcase only.
- LoRA or 8-bit Adam: not needed for memory.
- P1 by default (perf, product): contradicts the data policy [CR G14].

**Consequences.** Two languages in the training path, but Rust owns everything that can cause skew.

**Validation gate.** The trainer validation above; the M0 throughput gate C1; T9.

**Evidence.** [RT §0, §3, §4.4–4.5, §7, §8]; [CR G4, G13, G14, C7]; [LTR §4.5, §10]; review panels (A7, critical disagreement 8).

---

### ADR-024: Data licensing policy, data pools and contamination control

**Status:** Accepted. The share-alike part is Proposed-needs-user-input (Q4); the typed-decisions part depends on Q3. **Area:** A7.

**Context.** The proposals disagreed, and two of them contradicted themselves:
- Perf and product trained on Banking77 (and product on GoEmotions) while comparing against Jev on suites built from Banking77.
- Accuracy trained on MASSIVE-train while calling MASSIVE held out.
- Civil Comments text is CC-BY-SA, MultiNLI's fiction genre contains a CC-BY-SA work, and Super-NaturalInstructions has per-task licences [RT §6.1].

**Decision.**

*Licence gate:*
- Training allowlist: Apache-2.0, MIT, BSD, CC0, CC-BY, ODC-BY, CDLA-Permissive.
- CC-BY-SA and CDLA-Sharing are **excluded** unless the maintainer decides otherwise (Q4).
- NC or unknown licences are evaluation-only.
- Every source has `data/manifests/<source>.toml` with URL, revision, sha256 per file, SPDX id, `allowed_use`, and split policy.
- `arbitro-data` refuses to pack any source without `train` in its `allowed_use`.
- Each run writes a `data.lock`.
- The licences in [RT §6.1] are REPORTED, so every card is re-checked at ingestion with a human checklist.

*Four pools.* T is disjoint from O ∪ J ∪ L at source level; O and J are disjoint at source level. Item-level MinHash/13-gram checks of T against O ∪ J ∪ L run at build time. The final assignment is frozen in `evals/registry.toml` in M3a.

| Pool | Use | Initial members |
|---|---|---|
| **T** (train) | Training and teacher labelling | CLINC150 (out-of-scope → none); MultiNLI with the fiction genre removed until the manifest check clears it; PAWS; HellaSwag; GoEmotions; Aegis 2.0; HelpSteer2/3 (score); Super-NaturalInstructions classification tasks with explicit label sets (per-task licence filter); MultiWOZ (if MIT is verified); targeted synthetic sets (≈ 30 %, ADR-025) |
| **O** (OOD-S) | Held-out-source evaluation; dev for selection and calibration, test read once per release | MASSIVE-en (intents: validation split as dev, test split as test); ANLI (NLI; NC is fine for eval); toxic-chat and deepset prompt-injections (safety); SST-5 (score); ARC-Challenge (multiple choice; SA is fine for eval); probes (ADR-027) |
| **J** (Jev-comparable) | Test-only; never used for selection, calibration or synthesis | Banking77 (DMB S1, BTZSC-72); AG News and DAIR emotion (BTZSC); SMS spam (DMB S2); DMB S3–S5; tweet_topic, fin_topic and daily_dialog (elcronos); MMLU / MMLU-Pro and WANLI (Kev transfer suites); JevBench v1.2 public; priorbench 400; yibie 40 |
| **L** (Laya reproduction) | Runtime correctness only | MASSIVE test in 51 languages (Laya protocol); Laya T4 suites (seed 13); feishu_zh; typed-decisions test |

*The exclusion list* (never in pool T): Banking77, AG News, DAIR emotion, SMS spam, tweet_topic, fin_topic, daily_dialog, MMLU/MMLU-Pro, WANLI, **all of MASSIVE**, typed-decisions, and Tobi-Bueck/customer-support-tickets (CC-BY-NC).

*Contamination tags on every result row:*
- `in-train`;
- `held-out-source`, which means the family is seen in training. Examples: DAIR emotion, because GoEmotions trains the emotion family; Banking77 and MASSIVE, because CLINC150 trains intents;
- `held-out-family`.

The model card also discloses the backbones' unknown pretraining exposure.

*typed-decisions* is evaluation-only [CR G14], except for the private P1 if Q3 = yes. Suites without a licence file (DMB, priorbench, elcronos) are fetched by commit hash at run time and never vendored.

*Multilingual data* (M8) follows the same rules and is decided in M7.

**Alternatives.**
- Training on the comparison datasets: makes every "vs Jev" number in-distribution against Jev's zero-shot.
- Accuracy's MASSIVE-train: contradicts using MASSIVE as held-out.

**Consequences.**
- We give up some in-domain accuracy on the comparison suites in exchange for honest comparisons.
- The intent family trains on CLINC150 plus synthetic data only.

**Validation gate.**
- The manifest gate runs in `arbitro-data`.
- A CI overlap report (MinHash/13-gram) runs on every data build.
- G-Q hygiene (ADR-022).

**Evidence.** [RT §6.1, §8.5]; [BW §1.3, §4.2]; [CR G14]; [P-acc A7.3, A8.1]; review panels (A7, A8, critical disagreement 7).

---

### ADR-025: Teachers, soft labels and targeted synthesis

**Status:** Accepted. Each teacher licence is re-verified at use. **Area:** A7.

**Context.**
- Gold labels are missing or noisy for many decision framings.
- Teacher bias is family-specific.
- The terms of Gemma, of Llama and of hosted APIs restrict training on their outputs [RT §6.2].
- Aiming synthesis at an evaluation suite's family list is teaching to the test.

**Decision.**

*Allowed teachers* (Apache-2.0 or MIT weights, quantised so they fit in 24 GB):

| Teacher | Format on the 4090 |
|---|---|
| Qwen3-8B | bf16 or FP8 |
| Qwen3-30B-A3B | **4-bit AWQ/GPTQ only**, because FP8 (≈ 30 GB) does not fit |
| gpt-oss-20b | MXFP4, ≈ 13 GB |
| Phi-4 (MIT) | FP8 |
| Mistral-Small 3.x | 4-bit |
| OLMo 2 | as it fits |

*Excluded:* Gemma, Llama, every hosted API, Jev (including third-party Jev logs), and Laya outputs.

*Protocol:*
- vLLM on the 4090 with prefix caching.
- Score the log-probabilities of the option letters, averaged over ≥ 2 cyclic option orders.
- Fit the teacher's temperature on gold dev data first.
- Ensemble two teacher families by averaging log-probs.
- **Teacher gate:** a teacher labels a family only if it beats the current student on that family's OOD-S dev.
- Mix with gold as t = α·gold + (1−α)·teacher (α = 0.8), and only where gold is missing or noisy.
- Throughput: 1M decisions ≈ 300M prefill tokens ≈ 10–17 GPU-h per teacher (C4).

*Targeted synthesis* (decode-bound, 10k–100k items). Families are derived **only** from:
- the failure taxonomy in [BW §3]: traps and decoys, long policies with explicit day counts, explicit-date temporal items, multi-hop records, near-duplicate catalogues, assertion-style noul, evidence-removed unknowables, high-cardinality catalogues, multi-intent;
- our own OOD-S dev error analysis.

They are never derived from pool-J items, templates or family lists. The generator specifications are committed *before* any pool-J result is viewed, and the MinHash check against O ∪ J runs on all synthetic output.

*Model card:* discloses every teacher and synthesis family.

**Alternatives.**
- Accuracy's "aim at the JevBench hard-tier families": rejected as teaching to the test.
- Single-teacher labels: family bias.

**Consequences.** Some synthetic families necessarily resemble JevBench categories (long policies are a real use case). The JevBench tag therefore says the family was seen (ADR-022).

**Validation gate.**
- Teacher-vs-gold agreement on dev is reported per family.
- The overlap report.
- The teacher gate log in `reports/`.

**Evidence.** [RT §6.2–6.3]; [LTR §10.8]; [BW §3]; [JAS §8]; review panels (A6, A7).

---

### ADR-026: GPU operations and compute budget

**Status:** Accepted. The budget is re-based on the M0 throughput measurement. **Area:** A7.

**Context.**
- One 4090 is the development machine, the CI runner, the trainer and the labeller.
- All training throughputs are modelled (C1).
- Perf's total did not follow from its own line items.

**Decision.**

*Schedule:*
- The **GPU night** runs about 23:00–08:00. The nightly CI job runs first (≈ 45 min), then training or labelling.
- Days are for engine development and benchmarks. We never benchmark while training.
- Training uses a power limit of 350–380 W (C6). Benchmarks run with clocks locked and the power limit recorded.

*Resumability:* every job is resumable. Checkpoints carry the optimiser state, the RNG and the Rust sampler cursor.

*Run directories and tracking:*
- Runs are local-first: `runs/<date>-<slug>/` holds `config.resolved.toml`, `data.lock`, `metrics.jsonl`, `eval/*.json`, `MANIFEST.json` and `ckpt/`.
- MLflow is optional.
- Release-run summaries are committed to `reports/`.

*Budget to v0.3* (ESTIMATED at C1):

| # | Job | GPU-h |
|---|---|---|
| 1 | M0 spikes, PyTorch baselines, golden generation | 4–6 |
| 2 | E1 early backbone and layout signal | 10–15 |
| 3 | M3a teacher labelling (2 teachers × 1M decisions ≈ 600M prefill tokens) | 20–35 |
| 4 | M3b trainer validation (+ P1 if Q3 = yes) | 1–2 |
| 5 | M5 ablations X1–X7 (≈ 5B tokens) | 55–80 |
| 6 | M6 final training: `arbitro-en-large` 3 seeds (18–28) + `arbitro-en-base` 3 seeds (7–10) | 25–38 |
| 7 | M6 calibration, OOF correctness head, evaluation | 3–5 |
| 8 | Engine benchmarking and nightly CI over the period | 8–12 |
| | **Total** | **126–193** |
| | **With 25 % retry slack** | **158–241** |

After v0.3:
- M8 multilingual: 20–40 GPU-h (ESTIMATED, including translate-train).
- M9 FP8 calibration tables: about 5 GPU-h.

*Re-plan rule:* if M0 measures < 20k tok/s for ModernBERT-large (C1), the budget is scaled linearly and the milestone plan is re-baselined before M3a.

*Disk:* ≥ 1 TB recommended for datasets, teacher caches and checkpoints (Q6).

**Alternatives.** A cloud-GPU burst. It is not needed at these sizes; see Q12.

**Consequences.** About 3 weeks of GPU nights to v0.3 at the high end, spread over months.

**Validation gate.**
- The M0 throughput report.
- A per-milestone GPU-h actuals vs plan table in `reports/`.

**Evidence.** [CR C7, C13]; [RT §4–§5, §6.2, §8.3–8.4]; [LTR §9]; [P-prod A7], [P-acc A7.5]; review panels (A7).

---

### ADR-027: Evaluation suites, statistics and honest reporting

**Status:** Accepted. **Area:** A8.

**Context.** Laya's headline claims mixed several kinds of number without saying so:
- in-distribution datasets presented alongside zero-shot ones;
- ECE after a refit compared with other systems' raw ECE;
- runtime results and model results in one table.

See [BW §0.2, §1.4]. Kev's practice of verifying claims in CI prevents this kind of drift [RT §8.4].

**Decision.**

*Suite tiers* (`evals/registry.toml`, with dataset, revision, sha256, licence, pool, contamination tag and split; items are hashed and frozen):
- **E-T0 parity:** L0–L5 (ADR-014).
- **E-T1 Laya reproduction:** pool L.
- **E-T2 OOD-S:** pool O. Its test macro accuracy is the primary endpoint.
- **E-T3 Jev-comparable:** pool J, test-only. It runs under each suite's original protocol (items, option rendering, instructions).
- **E-T4 probes:**
  - permutation flip rate (all permutations for k ≤ 4, 3 random permutations otherwise);
  - label-swap consistency; shuffled context; empty-state prior; unknowable share at ≥ 0.9 confidence;
  - paraphrase consistency and noul negation;
  - cardinality 2…255; chunk invariance;
  - truncation sensitivity and long context 512…8k;
  - question isolation;
  - per-language accuracy with the router in the loop.

*Metrics.* One canonical implementation in `arbitro-eval`, unit-tested against hand-computed cases:
- accuracy, and macro-F1 over stable label texts;
- Brier, and NLL with a declared floor;
- ECE-15, both equal-width (with a documented bin-0 convention and a Laya-compatible flag) and equal-mass;
- classwise ECE and the rate of gold p = 0;
- AUROC, AURC, and coverage at ≤ 5 % error (threshold chosen on dev);
- MAE, within-1 and RPS for score;
- TV and KL against soft labels.

Calibration is **always reported three ways**: raw (T = 1), shipped, and held-out refit. A refit ECE is never compared with another system's raw ECE.

*Statistics:*
- paired, record-clustered bootstrap (cluster = state id), ≥ 2,000 resamples, fixed seed;
- McNemar's exact test for paired accuracy;
- one pre-registered primary endpoint per release, with everything else secondary;
- 3 seeds for model claims;
- failures counted as errors, never dropped;
- per-case JSONL committed for every published number (as release assets if large).

*Reporting rules, enforced in CI:*
1. **Two tables, never merged.**
   - "Runtime": same weights on a different engine; parity and speed only.
   - "Models": dm2 vs Laya (run by us, same harness) vs open baselines (TF-IDF + LR at 0.661, zero-shot NLI, a Qwen3 logprob teacher) vs third-party published Jev numbers.
   - laya-plus rows sit in the runtime table and are labelled as such.
2. **Contamination tags** on every row (ADR-024).
3. **Rendered numbers.** README and documentation numbers are rendered from `reports/*.json` by `cargo xtask numbers`. Every claim appears in `reports/claims.toml`, and `arbitro eval verify-claims` fails CI on any drift.
4. **Jev numbers** are only ever cited from third-party publications, with source, date, n and protocol differences, plus the note "Arbitro authors did not access the Jev API". They never enter selection or calibration.
5. **typed-decisions** is reported as an "in-distribution-teacher benchmark", never as zero-shot quality. Jev's 0.727 on it is not cited, because its provenance is circular [CR §3 #7].
6. **Test reads** are logged (ADR-022).

*Meta-tests:*
- the metrics crate reproduces Laya's published T4 ECE and Brier from Laya's own per-case outputs;
- `ece_score` passes its two hand cases [LIS §13 #7].

**Alternatives.** Laya-style single tables: rejected.

**Consequences.** Every published number needs a report file. Writing documentation is slower, but the numbers are trustworthy.

**Validation gate.**
- The claims check and the metrics meta-tests run on every PR.
- The E-T2/E-T3 runs happen at release candidates.

**Evidence.** [BW §0.2, §1.4, §4.1–4.4, F18]; [RT §8.4–8.5]; [CR §3 #7]; [P-acc A8], [P-prod A8], [P-perf A8]; review panels (A8).

---

### ADR-028: Performance benchmarking methodology and performance gates

**Status:** Accepted. **Area:** A8.

**Context.**
- Closed-loop load generators hide queueing: a slow response delays the next request, so tail latency is under-reported (coordinated omission).
- Clock boost and power limits move results by 5–10 %.
- Every 4090 number is modelled until measured (§5.1).

**Decision.**

*`arbitro bench` workloads* are checked-in files:
- questions per call ∈ {1, 5, 10, 50, 255};
- lengths ∈ {64 … 8k};
- options ∈ {2 … 255};
- mixed-length traffic;
- distinct vs repeated states;
- the P1–P4 and P9 workloads exactly as defined in §5.1.

*Procedure:*
- Warm up, then run ≥ 30 s of steady state (≥ 1,000 requests).
- An **open-loop Poisson** arrival generator, at arrival-rate and concurrency sweeps of {1, 4, 8, 16, 32, 64}.
- Report p50, p95, p99 and p99.9, plus throughput.
- Segmented breakdown from `server-timing`: tokenise, queue, GPU, post-process.
- A separate nsys profiling run.

*Conditions:*
- GPU clocks locked with `nvidia-smi -lgc`, power limit recorded.
- J/decision measured via NVML.
- Every report states the GPU, driver, CUDA, CPU and threads, precision, determinism mode, graphs on/off, batch policy, exact token counts, and git SHA.
- The baseline is the Laya PyTorch reference on the **same** machine, measured in M0 and written to `docs/perf-baseline.md` as "the number to beat".

*Nightly regression gate:* P13.

*Release performance gates:*
- v0.1: P6 and P7.
- v0.2: P1–P4 and P10 (reported).
- v0.3: P9.

**Alternatives.** Closed-loop-only load generation: rejected.

**Consequences.** The GPU needs a quiet window for benchmarks (ADR-026).

**Validation gate.** `reports/perf.json` for each release, and the nightly regression job.

**Evidence.** [BW §4.4]; [RIS §5]; [P-perf A8], [P-acc A8.5], [P-prod A8]; review panels (A8).

---

### ADR-029: CI topology and the self-hosted runner

**Status:** Accepted. Q5 sets the platform matrix; Q6 approves the runner. **Area:** A8/A11.

**Context.**
- CI must stay green for a solo maintainer.
- The only GPU is the maintainer's 4090.
- A self-hosted runner that executes code from fork PRs is a security hole.

**Decision.**

| When | Where | Checks |
|---|---|---|
| Every PR | GitHub-hosted Linux x86_64 | fmt; clippy `-D warnings`; tests; `cargo-deny`; `cargo-audit`; MSRV build; L0, L1, L2 (CPU), L5 (tiny model); pycompat property tests (10k); SDK conformance; claims check and metrics meta-tests; docs build; tiny-model training smoke test (Python); the `--no-default-features` build |
| Every PR (Tier-2) | GitHub-hosted macOS arm64 | Build (including `metal`), CPU tests |
| Weekly | GitHub-hosted Windows x64 (Tier-3) and Linux aarch64 (Tier-2) | Build only; CPU tests on aarch64 |
| Nightly (self-hosted 4090; `main` and scheduled jobs only, **never fork PRs**, no secrets) | The maintainer's machine, first slot of the GPU night | L2 on CUDA; L3 on real weights (CPU fp32 + GPU bf16, T3–T5); the bitwise batch-invariance test; perf regression (P13); fp16 and fp8 max-abs sweeps; pycompat property tests (10⁶); 10-minute training smoke run |
| Weekly | Self-hosted runner | Regenerate the reference goldens and diff them |
| Release candidate | Self-hosted runner + hosted | L4; the release performance gates; for model releases, E-T1…E-T4 and G-Q1…G-Q6; `cargo xtask release-check` (NOTICE and THIRD_PARTY_LICENSES regenerated; no `*.safetensors` in images; SBOM; signatures) |

*Where the builds happen:*
- The `-cuda` Docker image and the CUDA binaries are built on the self-hosted runner in release jobs triggered by a tag on `main`.
- The `-cpu` image is built on hosted runners.

**Alternatives.** Windows on every PR (product): avoidable maintenance.

**Consequences.**
- GPU regressions surface the next morning, not in the PR.
- Contributors cannot trigger GPU jobs.

**Validation gate.** The runner configuration is documented in `docs/ci.md`, and a quarterly check confirms that fork-PR workflows cannot target the self-hosted label.

**Evidence.** [P-prod A8, A11], [P-perf A8]; review panels (A8, A10, A11).

---

### ADR-030: Licensing, third-party code and supply chain

**Status:** Accepted. Q2 answered 2026-09-24: Apache-2.0 only. Counsel items (Q8, Q13) remain open. **Area:** A9.

**Context.**
- We vendor BSD-3 code (FA2, CUTLASS).
- We may vendor Apache-2.0 code (vLLM's `scaled_mm` templates).
- We port Apache-2.0 behaviour (Laya).
- CUDA libraries and MKL have their own redistribution terms.

**Decision.**

*Project licence:*
- **Apache-2.0** for all code, documentation and our model weights.
- DCO sign-off, not a CLA.
- `arbitro-cuda` is declared `Apache-2.0 AND BSD-3-Clause` because of the vendored kernels.

*Third-party code:*
- Lives in `third_party/<name>/` with its original LICENSE and a `MODIFICATIONS.md`.
- `third_party/README.md` records file-level provenance: upstream repository, commit, and path.
- The root `NOTICE` lists **copied code only**:
  - the FA2 hdim64 kernel files (BSD-3, Tri Dao);
  - CUTLASS, vendored at a pinned tag (BSD-3, NVIDIA);
  - vLLM's `scaled_mm` sm89 templates (Apache-2.0), when vendored in M9;
  - the Laya-derived logic files ("behavioural port of Laya © Convai Innovations, Apache-2.0").
- The TEI serving *architecture* is credited in the documentation acknowledgements. If any TEI code is ever copied, it goes into NOTICE.
- There is no build-time fetch of CUTLASS in our own crates. The `candle-flash-attn` dependency of `candle-cuda` fetches it; release and image builds pre-seed a sha256-checked checkout instead (Q16).

*Dependencies:*
- The `cargo-deny` allowlist is Apache-2.0, MIT, BSD-2/3, ISC, Unicode-3.0 and Zlib; anything else fails CI.
- `cargo-about` produces `THIRD_PARTY_LICENSES.html` in every artifact and image.

*Supply chain:*
- `SECURITY.md` (private advisories).
- `cargo-audit`.
- An SBOM via `cargo cyclonedx`.
- Signed artifacts (GitHub attestations).

*NVIDIA:*
- Images use the official `nvidia/cuda` runtime base images.
- Binaries load CUDA dynamically.
- The redistributable list needs counsel review (UNVERIFIED).

*MKL:* never linked statically without counsel. Only a dynamic, optional feature.

*Models:*
- Our weights are released as Apache-2.0 only after the licence gate passes. Each release carries a model card with the data manifest, the teacher disclosure and the eval report.
- mmBERT-derived multilingual weights are blocked until counsel reviews the Gemma-2-derived tokenizer (Q8).

*Fixtures:*
- Committed golden fixtures are logits and ids produced from Laya weights. They are test outputs, kept minimal, and documented in NOTICE and `docs/clean-room.md` as never used for training.
- Laya weights are never in the repository, in images or in release assets.

**Alternatives.** MIT OR Apache-2.0: vendored and derived parts would still be Apache-only, which makes a messy licence story.

**Consequences.** The release checklist carries licence items.

**Validation gate.**
- `cargo-deny` on every PR.
- `cargo xtask release-check` asserts: NOTICE and THIRD_PARTY_LICENSES regenerated; `docker run … find / -name '*.safetensors'` is empty; SBOM present; signatures present.

**Evidence.** [RIS §2.3, §4.6]; [LTR §5.2]; [CR G14]; [RT §6.1–6.2]; [P-prod A9], [P-perf A9], [P-acc A9]; review panels (A9).

---

### ADR-031: Marks, clean room and terms-of-service hygiene

**Status:** Accepted. **Area:** A9.

**Context.**
- TypeSafe's MCA §2.3(b) forbids using the service or its output to build a competing product.
- MCA §2.3(f) forbids publishing benchmarks of the service.
- Both are REPORTED [JAS §8].
- The wire format itself is derived from MIT-licensed SDKs and the public OpenAPI.
- Laya's training mix includes CC-BY-NC data [LTR §5.2].

**Decision.**

*Nominative use only.* "Jev", "TypeSafe", "System One" and "Laya" are used only to describe compatibility, for example "compatible with the TypeSafe Jev API wire format" and "runs Laya checkpoints". No logos. The README carries the disclaimer "Not affiliated with or endorsed by TypeSafe AI or Convai Innovations."

*Identifiers:*
- The marks never appear in project, crate, binary, image, mode or method names, config keys, or our model ids.
- **Exempt** because the wire protocol requires them:
  - the route `/v1/systemone`;
  - the header `x-typesafe-request-id`;
  - the client-side variable `TYPESAFE_BASE_URL` (documentation only);
  - the alias *values* `jev-latest` / `jev-preview`. They are accepted as input, unlisted by default, and can be disabled.
- The optional `/typesafe/v1/*` path alias is off by default.
- The Laya descriptors (`laya` mode, `laya-v1`, `laya-*` registry ids, `arbitro.compat.laya`) are nominative references to an Apache-2.0 project.

*Clean room:*
- `docs/clean-room.md` exists from week 1. It records the sources: the MIT SDKs, the public OpenAPI (as mirrored), third-party public data used for response shapes only, and Laya's Apache-2.0 source.
- CONTRIBUTING rule: **"Do not use a TypeSafe account to develop, test, or benchmark this project."**
- We never produce Jev numbers ourselves.
- Third-party Jev logs are referenced by URL and commit, used for shape tests only, never vendored, and never used as labels, for calibration or for selection.

*Laya:*
- Weights are never redistributed.
- Weights are never used to initialise our models.
- We never distil from Laya outputs.
- Loading a user-downloaded Laya checkpoint at runtime is fine.

**Alternatives.** Product's `jev` / `jev-strict` modes and `system_one` method: rejected.

**Consequences.** If TypeSafe objects, the aliases can be disabled. The request-id header must stay, because the Python SDK raises without it.

**Validation gate.**
- A CI grep lint rejects `jev|typesafe|system_?one` in crate names, config keys and public API identifiers, outside an allowlist.
- A PR-template checkbox for the account rule.

**Evidence.** [JAS §8, §9.4 #2, #12]; [LTR §5.2]; [P-perf A9], [P-acc A9], [P-prod A9]; review panels (A9).

---

### ADR-032: Roadmap and capacity plan

**Status:** Accepted. It is re-baselined at each milestone exit, and after the M0 measurements. **Area:** A10.

**Context.**
- A solo developer cannot parallelise their own hours. Only GPU nights run in parallel with coding.
- Perf's plan was over-committed (455 h in 30 weeks).
- Accuracy's plan delayed the engine.
- Product's estimates were about 1.5× optimistic for M2 and the engine.

**Decision.** 12 h/week. Cumulative hours set the week numbers.

| Milestone | Dev-h | Cum. | ≈ Weeks | GPU-h | Deliverables | Exit gate |
|---|---|---|---|---|---|---|
| **M0** Foundations & measurement | 24 | 24 | 1–2 | 4–6 | Workspace, CI, LICENSE/NOTICE/DCO, `docs/clean-room.md`; 4090 bring-up; the pins, safetensors header check and sha256s; the PyTorch baseline; spikes (training tok/s, CPU backend, cuBLASLt/FA2, fp16 headroom); `tools/goldens`; start of `pycompat` | Four spike reports; C1 measured; L0 `pycompat` green; L1 ≥ 1k items; Q1, Q2, Q6, Q7 answered or defaulted |
| **M1** Compat core + CPU runtime | 56 | 80 | 3–7 | in #1 of ADR-026 | `arbitro-proto/core/compat` (pycompat, render, validate, sequence, temps, post), tokenizer wrapper, registry and loader, `arbitro-candle` `cpu`, `arbitro decide`/`pull` | L0–L3 green for all three checkpoints on `cpu` fp32 |
| **E1** Early model signal (interleaved) | 18 | 98 | 5–11 | 10–15 | A spike trainer (packed ModernBERT, layouts L0 and L2), a gold-only mini-mixture, 4 backbone arms, layout L0 vs layout L2 | ADR-020a signal recorded |
| **M2** v0.1 "drop-in" | 60 | 158 | 8–14 | in #8 of ADR-026 | `arbitro-server` (3 modes, errors, auth, limits, metrics); `candle-cuda` bf16; conformance suite; `-cpu` and `-cuda` images; cargo-dist binaries (Linux x86_64, macOS arm64); mdBook quickstart and migration guide | **v0.1 definition of done** (ADR-001) → **release v0.1 (~week 14)** |
| **M3a** Data & teacher pipeline | 24 | 182 | 15–16 | 20–35 (nights, during M4) | Manifests + licence gate, pool assignment, converters for mixture v1, MinHash check, vLLM labelling scripts | Overlap report clean; labelling running |
| **M4** Custom CUDA engine | 170 | 352 | 16–30 | in #8 of ADR-026 | `arbitro-cuda`: K1–K11 and the debug fp32 path (K13, Q17), cuBLASLt raw API, FA2 vendoring, arena, piecewise then full graphs, batch invariance, fp16 mode, `arbitro bench` | T4, T5, P1–P4, bitwise test |
| **M4c** Compat completion | 12 | 364 | 30–31 | — | `lang` + the `laya-heuristic` router (LIS #30–47, #63) | L0/L4 router goldens → **release v0.2 (~week 31)** |
| **M3b** Trainer, calibration & eval foundation | 46 | 410 | 31–35 | 1–2 | Production trainer, PyO3 data path, `calib_fit`, `arbitro-eval` suites/statistics/claims, export parity, trainer validation (+ P1 if Q3 = yes) | Trainer validation (ADR-023); T9 on the tiny model |
| **M5** Ablation programme | 35 | 445 | 35–38 | 55–80 | X1–X7 pre-registered and run; ADR-019a / ADR-020a frozen | All decisions made with CIs |
| **M6** First own model | 60 | 505 | 38–43 | 28–43 | dm2 frontend (chunking, none option, dual channel); the K3b paged path if layout L2 is adopted; `arbitro-en-large` / `-base` (3 seeds); calibration + conformal; model cards; eval report; public PyPI wheel | G-Q1…G-Q6, T9, P9 → **release v0.3 (~week 43)** |
| **M7** 1.0 hardening | 30 | 535 | 43–45 | ~2 | API freeze (`/v1`, `x_arbitro` v1), semver checks, full documentation, security review, SBOM and signing | Release checklist → **1.0 (~week 45)** |
| M8 Multilingual (only if Q8 clears) | 40 | 575 | 45–48 | 20–40 | `arbitro-multi-base`, `lid` routing, per-language evaluation | Per-language gates vs `laya-multilingual` → 1.1 |
| M9 FP8 opt-in | 30 | 605 | 48–51 | ~5 | K12, vLLM CUTLASS sm89 `scaled_mm`, per-precision calibration | P11 → 1.2 |

*First two weeks (M0, 24 h):*

Week 1:
- (2 h) Skeleton: `rust-toolchain.toml` pinned to 1.98.1, `deny.toml`, LICENSE, NOTICE, README (scope, non-goals, disclaimer), CONTRIBUTING (DCO, the account rule), SECURITY.md, Actions (fmt, clippy, test, deny).
- (3 h) 4090 bring-up (driver ≥ 580, CUDA 13):
  - `tools/goldens` uv environment;
  - fetch the three checkpoints at their pins and record sha256;
  - read the safetensors headers (F16?);
  - compare EN weights at `c5d78730` vs `1c5edc17`.
- (3 h) Microbenchmarks:
  - GEMM TFLOPS for bf16 / fp16-with-fp32-accumulate / FP8 at the model's (N, K) and M ∈ {256, 1k, 4k, 16k};
  - cuBLASLt bf16 → fp32 C/D with beta = 1;
  - FA2 hdim64 windowed varlen;
  - FP8 outer-vector probe (informational).
- (4 h) The PyTorch Laya baseline on the 4090: batch-1 latency at 250 tokens, saturated q/s, 5- and 50-question requests. The EN max-abs sweep. Write `docs/perf-baseline.md`.
- GPU night: the training-throughput spike, padded+checkpointing vs packed+compile, at L ∈ {128, 512, 1024} for ModernBERT-large, ModernBERT-base and mmBERT-base, plus DeBERTa-v3-large padded to 512.

Week 2:
- (4 h) `pycompat`: `json.dumps` in both modes, `repr(float)`, `round`, `%r`, with property tests against a Python subprocess.
- (3 h) Tokenizer wrapper, L1 generator, first `build_sequence` port.
- (2 h) CPU spike (ADR-007) → `reports/spikes/cpu.md`.
- (2 h) Golden generation: L3 seed (500 questions, CPU fp32 + CUDA bf16) and the tiny models.
- (1 h) The skeleton of `docs/ablations.md`; ADR files split from this document.
- GPU night: finish the L3 fixtures for 2k questions on all three checkpoints.

*Cut rules* (also in ADR-033):
- **C-1.** Any milestone > 50 % over its dev-h: re-plan at the next milestone boundary.
- **C-2.** Two milestones > 50 % over: drop M8 and M9 from the 1.x plan.
- **C-3.** M4 > 255 dev-h: `candle-cuda` stays the default GPU backend, M4c ships as v0.1.x, and the engine continues.
- **C-4.** DeBERTa wins X1: add a distillation round (+30 GPU-h, +10 dev-h) to M6.
- **C-5.** G-Q1 fails: the preview-release path (ADR-022).

*Sensitivity:*
- At 10 h/week, multiply the week numbers by 1.2.
- At 8 h/week, by 1.5: 1.0 lands at ~week 67.

**Alternatives.**
- Model-first ordering after v0.1 (M3a → M3b → M5 → M6, then M4): the first own model arrives at ≈ week 27 instead of ≈ 43, served on `candle-cuda` with the layout L2 gather-then-dense path. The custom engine then slips from ≈ week 31 to ≈ week 43. Rejected as the default because both review panels prioritised the performance headline, and because E1 already de-risks the backbone early. It is a pure re-ordering with no redesign, so the maintainer can choose it (Q14).
- Engine and model in parallel: impossible for one person.

**Consequences.**
- v0.3 lands at about month 10, later than the review panels' "month 8–9". The difference is E1 (+18 h) and the compat completion (+12 h), both kept deliberately.
- GPU nights are productive during M4, because labelling runs then.

**Validation gate.** A milestone exit review writes `reports/milestones.md` with dev-h and GPU-h actuals vs plan.

**Evidence.** [P-prod A10], [P-perf A10], [P-acc A10]; [RT §7]; [CR C7]; review panels (A10).

---

### ADR-033: Risk register and cut policy

**Status:** Accepted. Reviewed at every milestone exit. **Area:** A11.

**Decision.**

| # | Risk | L / I | Mitigation | Early signal |
|---|---|---|---|---|
| R1 | **Solo bandwidth or burnout.** One maintainer carrying five deliverables | H / H | Tiers, non-goals, cut rules C-1…C-5, nothing Tier-2 blocks a release, self-checking "good first issues" | Milestone slip > 30 % |
| R2 | **ModernBERT-large does not learn from a cold start** [CR C9; kotoba] | **M-H / H** | E1 in weeks 5–11; ModernBERT-base and Ettin arms; warmup + LLRD + hybrid read-out; DeBERTa as the teacher → distillation (ADR-020) | E1 learning curves |
| R3 | dm2 does not clearly beat Laya, or stays far from Jev on hard zero-shot | M / H | Pre-registered ablations, teachers, targeted synthesis, the preview path, honest G-Q2 reporting; the runtime has value regardless | M5 X-results on OOD-S dev |
| R4 | The custom engine overruns its 170 h | M / H | laya-v1 subset first; piecewise graphs first; kernels added one at a time behind T4; C-3 | M4 burn-down at week 23 |
| R5 | Shared state (layout L2) costs accuracy | M / M | Layout L3-k; layout L0 fallback; the engine supports layout L0 natively | E1 layout L0-vs-L2 signal; X2 |
| R6 | Evaluation contamination makes the Jev comparisons dishonest | M / H | Pools, exclusion list, MinHash, tags, test-read log | CI overlap report |
| R7 | Legal: data licences, the Gemma-derived tokenizer, our trademark | M / H | Manifest gate, conservative defaults (Q4, Q8, Q13), English-first, counsel on 3 items | Ingestion review |
| R8 | The modelled 4090 numbers are wrong | M / M | M0 spikes; gates re-based and recorded | M0 |
| R9 | FP8 per-token/per-channel scaling unavailable in cuBLASLt on Ada | **H / L** | CUTLASS sm89 is the plan, not a fallback; FP8 is post-1.0 anyway | M0 probe |
| R10 | Jev contract drift (young SDKs, no changelog) | M / L | Pinned SDK conformance + Renovate | A red conformance test |
| R11 | Laya reference or Hub drift | H / L | Pins + sha256; compat targets 0.3.7, installed from git by commit (0.3.7 left PyPI on 2026-09-24, AM-17); weekly regeneration | Nightly L3 |
| R12 | One 4090 is dev machine, CI runner and trainer at once | H / M | The GPU-night schedule; CPU CI covers PRs | Queue conflicts |
| R13 | Calibration does not transfer out of distribution | M / M | Feature-conditioned T, correctness head, conformal; the OOD-S dev gate | G-Q3 on dev |
| R14 | Option budgets overflow (255 options × spans) | M / M | Option-group chunking + random chunking in training; the chunk-invariance gate | X6 |
| R15 | CPU speed spoils first impressions | M / M | MKL/Accelerate or ORT (ADR-007); `arbitro-en-base`; honest `auto` limits | Week-2 spike |
| R16 | Self-hosted runner security | L / H | `main` and scheduled jobs only, no fork PRs, no secrets (ADR-029) | Quarterly audit |
| R17 | Bus factor | H / M | Docs-as-code, reproducible `xtask`, ADRs, fixtures | — |
| R18 | Prior art moves (`laya` 0.1.1, `laya-rs` 0.1.0, a Convai Rust server) | M / M | Differentiate on the Jev contract, parity, CUDA speed and own models; offer `pycompat` upstream (Q10) | Watch the repos |
| R19 | Batch invariance too expensive or unattainable | M / L | `fast` mode; a post-1.0 fixed-K GEMM; the "designed to be" wording until proven | P10 in M4 |
| R20 | CUDA build and distribution (nvcc, versions, image size) | M / M | Prebuilt binaries and images, dynamic CUDA loading, hdim64-only FA2 | Image > 2 GB, or a build > 30 min |

*Cut rules:* C-1…C-5, as in ADR-032.

*Open questions:* §2 (Q1–Q18). An unanswered question proceeds on its default. Each answer is recorded in the relevant ADR, with a date.

**Evidence.** Every risk cites its ADR. Review panels (A11): backbone risk raised to M-H/H; FP8 on cuBLASLt raised to likelihood H; contamination, burnout and option-overflow risks added.

---

## 8. Appendix A: review-panel findings and how each was fixed

| # | Finding (source) | Fix | Where |
|---|---|---|---|
| 1 | Perf: "≥ 450 q/s … ≥ 10× PyTorch" is unsupported | No 10× claim anywhere. P5 is ≈ 2.5× throughput; batch-1 gains are measured in M0. | §5.1 P5, ADR-001 |
| 2 | Perf: 0.1 in 16 weeks needs 225 + 50 h; 455 h in 30 weeks does not add up | Hours re-derived and serialised | ADR-032 |
| 3 | Accuracy: `rustify-v2-xl` is scope creep | The decoder tier is a non-goal before 1.0 (Q12) | ADR-001 |
| 4 | Product: v0.1 in 40 h is optimistic | M2 = 60 h, with trimmed platforms | ADR-032 |
| 5 | Product: a single `1c5edc17` pin | Per-checkpoint, per-file pins | ADR-005, §3.5 |
| 6 | Product: `PackedBatch` lacks `position_ids`; `SharedPrefix` is too narrow | Generalised `PackedBatch` | ADR-004 |
| 7 | Accuracy: 18 crates; perf's 12 vs 14 miscount | 8 published + 3 internal | ADR-003 |
| 8 | Accuracy: "Rust 1.94" presented as the toolchain | Toolchain 1.98.1 (VERIFIED), MSRV 1.96 | ADR-003 |
| 9 | Perf: `tydec` is taken on npm | Recorded; Arbitro recommended | ADR-002 |
| 10 | Perf: "2 of 71 FA2 files"; the paged path needs the splitkv kernels | Correct count is 53 `.cu` files. K3 uses the dense hdim64 files; K3b vendors `flash_fwd_splitkv_hdim64_{bf16,fp16}_sm80.cu` | ADR-008 |
| 11 | Perf: fp32 C/D and fp32 GELU/scorer described as "bit-closest" | Re-described as closer to fp32 than the reference; T4 budgets for the reference's bf16 quantisation | ADR-008, ADR-014 |
| 12 | Perf: the L4 2e-3 max target | Mean ≤ 2e-3 / max ≤ 2e-2 (T4) | §5.4 |
| 13 | Product: `max_wait_us = 1500` | Zero-delay, `max_wait_us = 0` | ADR-010 |
| 14 | Accuracy: custom engine at M5 without a technical dependency | The engine follows v0.1 | ADR-006 |
| 15 | Accuracy: act-head features "on the host" | K10 runs on the GPU with the CLS row | ADR-008 |
| 16 | All: FP8 via cuBLASLt outer-vector scales on sm_89 | CUTLASS sm89 is the planned route | ADR-009 |
| 17 | Perf: CPU "≤ 0.8× PyTorch" | Gate ≤ 1.5×, goal ≤ 1.0× (P7) | ADR-007 |
| 18 | Perf: arena "168 MB" | 172 MB (164 MiB) (MEM6) | §5.2 |
| 19 | Product: engine at 80–84 h | 170 h budget + cut rule C-3 | ADR-032 |
| 20 | "sm_120 PTX" untested | Marked UNVERIFIED | ADR-008 |
| 21 | Perf: the `ensure_ascii=True` escaping of DEL and U+2028 | Escape table for both modes, VERIFIED | ADR-013 |
| 22 | All: the exact MASSIVE-51 gate is flaky | ±1 item on near-ties only (T8); argmax near-tie exemption (T3) | ADR-014 |
| 23 | Unicode 14 / CPython 3.11 vs Laya's Python ≥ 3.10 | Parity defined against 3.11 and documented | ADR-012 |
| 24 | Perf: router, email and presets on the v0.1 path | Router in v0.2; email and shortlist are community work | ADR-012 |
| 25 | Product: L0 described as "the 63 acceptance tests" | Mapping fixed: #1–29 / #30–47 / #48–54 are L0, #55–60 are L5, #61–63 are L4 | ADR-014 |
| 26 | Perf: 10 s deadline → 504 | 8 s → 503/529 | ADR-017 |
| 27 | Product: 4-decimal default with a 1e-4 floor | `full` with a 1e-6 floor | ADR-016 |
| 28 | Product: 429 when the queue is full | 503/529; 429 only per key | ADR-017 |
| 29 | Accuracy: strict mode counts the state once for laya-v1 | Processed tokens in all modes | ADR-018 |
| 30 | Perf: malformed JSON → 400 in lenient mode | 422 `json_invalid` | ADR-017 |
| 31 | Product: score confidence differs by mode, undocumented | `peak` by default in both modes; documented | ADR-016 |
| 32 | Perf: the tree layout wastes FA2 tiles; the 800-question target is not credible | Tree is an ablation arm only; no 800-question target | ADR-019 |
| 33 | Perf: a set head for score questions needs the "level i:" rendering | Made explicit | ADR-019 |
| 34 | Perf: "laya-v1 needs 1.7M tokens" is misleading | Removed; laya-v1 truncates at `max_len` | §5.1 |
| 35 | Accuracy: Kev's 0.837 → 0.852 misattributed | Joint attribution | ADR-022 |
| 36 | Accuracy: synthesis aimed at JevBench families | Families come from the failure taxonomy + OOD-S dev only | ADR-025 |
| 37 | Product: the dm2 latency gate is not falsifiable | P9 fixes the question length | §5.1 |
| 38 | Product: 255 × 33 tokens > 4,096 | Option-group chunking | ADR-019 |
| 39 | Product: no quantified bar for beating Laya | G-Q1 | ADR-022 |
| 40 | Accuracy: G-Q1 has no fallback; too many features for the budget | Preview path; v0.3 scope trimmed | ADR-022, ADR-019 |
| 41 | New: `laya-multilingual` cold start on mmBERT-base | Cited as counter-evidence (Q-ref12) | ADR-020 |
| 42 | New: distil a winning DeBERTa into a ModernBERT-family student | Adopted | ADR-020 |
| 43 | Perf and product: P1 on typed-decisions | Q3 opt-in; licence-clean validation by default | ADR-023 |
| 44 | Perf and product: Banking77 / GoEmotions / DAIR overlap; accuracy's MASSIVE contradiction | Pools, exclusion list, MASSIVE eval-only, tags | ADR-024 |
| 45 | Perf: GPU total not derived | Derived table (126–193 / 158–241) | ADR-026 |
| 46 | Perf: Qwen3-30B-A3B does not fit 24 GB in FP8 | 4-bit AWQ/GPTQ only | ADR-025 |
| 47 | All: Civil Comments (SA text), MultiNLI fiction, S-NI per-task licences | Excluded or filtered pending manifests | ADR-024 |
| 48 | Perf: Banking77-77 ≥ 0.80 is an in-distribution gate | Banking77 is eval-only; G-Q6 is relative to `laya-en` | ADR-022 |
| 49 | Product: `jev` / `jev-strict` modes and `system_one` | `strict`/`lenient`/`laya`, `decide` | ADR-015, ADR-031 |
| 50 | Product: NOTICE listed patterns, not copied files | NOTICE lists copied files only | ADR-030 |
| 51 | All: golden fixtures from CC-BY-NC-trained weights | Minimal, and documented as test outputs | ADR-030 |
| 52 | All: backbone risk under-rated | Raised to M-H/H with E1 | ADR-033 R2 |
| 53 | Perf: cuBLASLt risk rated L likelihood | Raised to H/L; CUTLASS is the plan | ADR-033 R9 |
| 54 | Perf, accuracy, product: missing risks (contamination, burnout, own-OOD contamination, option overflow) | Added: R1, R6, R14 | ADR-033 |

## 9. Appendix B: evidence keys

**Research reports.** Design-phase reports, dated 2026-09-23. They are not in the repository yet (Q15); the keys are kept so that every claim stays traceable. [ANALYSIS.md](ANALYSIS.md) summarises their findings on Jev and Laya, with links to the upstream sources.

| Key | Report | Content |
|---|---|---|
| [LIS] | `laya-inference-spec.md` | Behavioural spec of Laya 0.3.7 inference, 63 acceptance tests |
| [JAS] | `jev-api-spec.md` + `typesafe-ai-openapi.json` | Jev wire format, limits, errors, post-processing, terms (the OpenAPI document as publicly mirrored) |
| [LTR] | `laya-training-recipe.md` | RLCD reconstruction, fine-tune recipe, 4090 estimates, improvement list |
| [BW] | `benchmarks-weaknesses.md` | Quality bar, failure modes, latency data, eval-harness requirements |
| [EA] | `encoder-architecture.md` | ModernBERT/mmBERT/head architecture, tokenizers, numerics, FLOPs |
| [RIS] | `rust-inference-stack.md` | candle/ort/burn/TEI/cudarc evaluation, 4090 performance estimates |
| [RT] | `rust-training-4090.md` | Training-stack decision, memory and throughput, data strategy, licensing |
| [CR] | `critic.md` | Resolved contradictions; overrides the other reports |

**Proposals and review panels.** The three design proposals ([P-perf] `proposal-perf.md`, [P-acc] `proposal-accuracy.md`, [P-prod] `proposal-product.md`) and the verdicts of the two design-review panels ("review panels", areas A1–A11) are design-phase inputs. This record consolidates them and they are not published separately.

**Upstream sources**, cited by repository and path:
- `NandhaKishorM/laya` @ `010bacef` (Laya 0.3.7), e.g. `laya/common.py`, `laya/agent.py`;
- `EricYu123456/laya-hexagon-npu`, `models/` (a mirror of the Laya configs and tokenizers at `1c5edc17a7acd8701df6fc341c0d179f1c62c982`);
- `afshinm/laya-mps` (pins for `c5d78730…` and `f9ab0b22…`);
- `kotoba-lang/typed-decisions`, `README.md` ("kotoba README");
- `jaredpalmer/kev`, `README.md` ("kev README");
- `Heman10x-NGU/openJev-verdict-2.0` ("verdict2");
- the `candle-flash-attn` 0.11.0 crate (crates.io), e.g. `kernels/flash_api.cu`, `build.rs`, `src/lib.rs`.

**Checks made during the design phase** (2026-09-23):
- crates.io, PyPI and npm availability of `arbitro*`;
- Rust stable 1.98.1 (static.rust-lang.org);
- the CPython 3.11.15 / Unicode 14.0.0 / numpy 2.4.6 reference venv, and the `json.dumps` escaping behaviour;
- the FA2 hdim64 file list and the paged dispatch in `flash_api.cu`;
- the MASSIVE sweep metadata (laya 0.2.0, torch 2.8.0);
- the Ettin repository licence (MIT, code only);
- the SDK environment variable names (`TYPESAFE_BASE_URL`, default timeout 10 s).

---

## 10. Appendix C: amendments from the documentation review

The documentation consistency review of 2026-09-23 checked this record against ARCHITECTURE.md, ROADMAP.md, TRAINING.md, the README and the primary sources. It made these changes in place; the IDs AM-1…AM-17 are local to this appendix (AM-14 was added when ANALYSIS.md joined the documentation). The ADR files split into `docs/adr/` in M0 carry them.

| # | Where | Change | Reason |
|---|---|---|---|
| AM-1 | ADR-008 K10 | The act head's activation is exact-erf GELU, not ReLU | `nn.GELU()` at `laya/common.py:101` (VERIFIED); [EA §3.2] agrees |
| AM-2 | ADR-008 K7 | K7 also covers head layer 2's `norm1` | K6 fuses only layer 1's `norm1` |
| AM-3 | §5.2 MEM2 | 643.8 MB (614.0 MiB), not "≈ 615 MB" | 321,908,998 × 2 B; the mirror README's "615 MB" is the MiB value |
| AM-4 | §3.5, ADR-005, ADR-015 | The concrete id is returned in `strict` and `lenient` mode; `laya` mode returns `"laya-rl-agent"`; the `x-arbitro-model` header always carries the concrete id | Byte parity with laya-serve (`laya/agent.py:337, 427`) |
| AM-5 | ADR-009 | Only the bf16-GEMM weights are converted to bf16 at load; the other tensors stay fp16-exact and are widened to fp32 | The reference keeps non-GEMM tensors fp16-exact in fp32 |
| AM-6 | ADR-009, ADR-005 | `amp_dtype: "bf16"` was read at `1c5edc17`; M0 re-reads it at the `c5d78730` and `f9ab0b22` pins | [CR C1] read the mirror only |
| AM-7 | ADR-029 | "pyjson property tests" renamed to the canonical module name `pycompat` | ADR-013 |
| AM-8 | §1, ADR-019, ADR-020, ADR-022, ADR-032, ADR-033 | dm2 layouts written "layout L0/L2/L3/T" | Glossary rule (§4) |
| AM-9 | ADR-017, §3.3, ADR-001 | The Docker images set `ARBITRO__SERVER__BIND=0.0.0.0:8080` and `ARBITRO_HOME=/cache` | The v0.1 definition-of-done command could not reach a loopback bind |
| AM-10 | §2 Q15–Q18; ADR-006, 008 (K13), 010, 014, 017, 019, 030; §5.4 T5, T9 | Four new open questions with defaults: research-report publication, the CUTLASS fetch in `candle-flash-attn`, a debug-only fp32 GPU path for T5/T9, the admission cap | Gaps found by ARCHITECTURE.md (O-9, O-16, O-17, O-18) and ROADMAP.md |
| AM-11 | ADR-007 | No dev-h budget exists for `arbitro-ort`; the M0 exit estimates it if the spike fails | Found by ROADMAP.md |
| AM-12 | ADR-019, ADR-021, ADR-023 | Type tokens are inserted by id and marked special; shipped calibration artefacts are fitted only on training-licensed items; the loss floor's gradient is settled before M3b | Found by TRAINING.md (§4.4, §10.4, §9.1) |
| AM-13 | Header, Appendix B | Paths into the design environment replaced with upstream repository names; "judge panels" renamed "review panels" | The record is now part of the repository |
| AM-17 | ADR-012 reference environment; R11 | laya 0.3.7 is installed from git at `010bacef` because PyPI no longer serves it; R11 likelihood raised to H | PyPI JSON for `laya` on 2026-09-24 lists no 0.3.7/0.3.8; upstream HEAD is 173 commits past `010bacef` (git log) |
| AM-16 | §2 Q1/Q2, §3 names, ADR-002, ADR-030; README, ROADMAP | Q1 answered (Arbitro) and Q2 answered (Apache-2.0 only); ADR-002 accepted; LICENSE added | Maintainer decision, 2026-09-24 |
| AM-15 | ADR-016 `round2`; ARCHITECTURE.md §12 rounding row; §5.2 MEM1/MEM2 note | `round2` guarantees only the invariants seen in all logs; `choice` follows the unrounded argmax; the exact Jev rounding rule is marked unknown. MEM1/MEM2 count state-dict elements, which include the 3-element `temperature` buffer | ANALYSIS.md fact-check recomputed the DMB raw log (52 multi-entry artefacts; one 0.14-over-0.15 choice); `sum(p.numel() for p in model.parameters())` = 421,293,827 / 321,908,995 in torch 2.14 |
| AM-14 | Header, §3.6, Appendix B | ANALYSIS.md, the reference analysis of Jev and Laya, is listed as a companion document and in the repository layout | Written after the review; it quotes §5 values unchanged and is linked from the README and the companion documents |
