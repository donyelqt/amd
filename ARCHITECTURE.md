# Track-1 Token-Efficient Routing Agent — Architecture

> AMD Developer Hackathon: ACT II — Track 1. Local-first, zero-token routing agent.
> Profile `A81`. Image: `donyelqt/track1-agent` (Docker Hub + GHCR), `linux/amd64`.

## 1. Goal & core idea

The competition scores inference **only on Fireworks API calls made through the
harness-supplied `FIREWORKS_BASE_URL`**, using models from `ALLOWED_MODELS`.
Local in-process inference counts as **zero tokens** toward the final score
(after an accuracy gate). Therefore the agent's single objective is:

> **Answer every task correctly while spending as few proxy-recorded tokens as possible.**

The design is a **cascade of cheaper-and-cheaper solvers**, falling back to an
expensive remote LLM call only when nothing free can answer with confidence:

```
task ─▶ Tier 0: zero-token deterministic shortcuts ─▶ Tier 1: local GGUF (0 tok)
     │                                                    │
     │ (no free/confident answer)                         └─▶ Tier 2: remote Fireworks (costs tokens)
     └────────────────────────────────────────────────────────────────────▶ remote (with fallback)
```

## 2. Sacred invariants (`main.py:1-15`)

1. `/output/results.json` is **always** written, valid, and complete (every `task_id`).
2. The process **exits 0** whenever results exist.
3. A deadline guard flushes partial results instead of crashing.
4. All inference goes through `FIREWORKS_BASE_URL` with `ALLOWED_MODELS` ids only.

The orchestrator (`run()`, `main.py:1941`) is hardened so even a missing input
file, a dead remote, or an OOM still produces a well-formed results file and a
clean exit 0 — the harness never sees a crash.

## 3. Package layout

| File | Role |
|------|------|
| `main.py` | Agent core: routing, shortcut solvers, remote client, orchestrator, I/O. |
| `local_llm.py` | Thread-safe wrapper around `llama_cpp.Llama` serving the bundled GGUF in-process. |
| `Dockerfile` | Bakes `requirements.txt` + the GGUF into the image; `linux/amd64`; entrypoint `python main.py`. |
| `input/tasks.json` | Task set (50 tasks, ids `t1..t50`) for local validation. The harness supplies its own at eval. |

The image **bakes the local model at build time** (`Dockerfile:35-36`) because the
judging VM has no guaranteed internet. Default weight: Qwen2.5-3B-Instruct Q4_K_M
(~1.8 GB) for fast CPU inference well under the 10 GB image cap.

## 4. I/O contract & configuration

- **Input**: `/input/tasks.json` (overridable via `INPUT_PATH`). Reads a list of
  `{task_id, prompt}` or `{tasks:[...]}`.
- **Output**: `/output/results.json` (overridable via `OUTPUT_PATH`): a list of
  `{task_id, answer}` in task order, one row per task, never omitted.
- **Harness env vars** (`load_settings`, `main.py:1587`):
  - `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, `ALLOWED_MODELS` (comma-separated) — injected by the harness. **Never baked in.**
  - Optional: `CONCURRENCY`, `SOFT_DEADLINE_SECONDS`, `PER_CALL_TIMEOUT_SECONDS`,
    `LOCAL_MODEL_PATH`, `LOCAL_CONFIDENCE_GATE`, `AGENT_PROFILE`.

## 5. Router (`main.py:801`, `_route_category`)

A priority-ordered, regex-based 8-category classifier picks the solver path:

`exact_response → summarization → sentiment → ner → code_debug → code_gen → logic → math → factual`

Per-task output requirements (`parse_requirements`, `main.py:833`) extract word/sentence
limits, "exactly N" constraints, and justification requests so the answer honors them
(`enforce_limit`, `main.py:1860`).

## 6. The solver cascade (`Run.solve_task`, `main.py:1814`)

### Tier 0 — zero-token deterministic shortcuts (`solve_zero_token`, `main.py:1689`)
Pure-Python, no model, no network. Each returns an answer or `None`:
- `exact_response` — literal "reply with exactly …" instructions.
- `sentiment_shortcut` — keyword polarity scoring.
- `ner_shortcut` — regex entity extraction.
- `logic_shortcut` — deductive puzzles (tallest/oldest/…) via constraint propagation.
- `codegen_shortcut` — verified code templates (reverse, palindrome, fib, factorial, fizzbuzz, two-sum), executed and test-checked locally.
- `math_template_solve` / `solve_simple_math` — AST-evaluated arithmetic & percentage/discount/tip/interest templates.

### Tier 1 — local GGUF (`local_solve`, `main.py:1785`)
For `LOCAL_PRIMARY = {sentiment, factual, code_gen}` (`main.py:1640`): the bundled GGUF
(`local_llm.py`) answers **in-process → 0 Fireworks tokens**. On load failure or empty
output it transparently falls back to remote.

### Tier 1.5 — accuracy gate (`LocalVerifier`, `main.py:547`)
A confidence score per shortcut/local answer. If below `LOCAL_CONFIDENCE_GATE`
(default `0.6`), the answer is discarded and the cascade continues to the remote tier
instead of returning a confident-but-wrong free answer.

### Tier 2 — remote Fireworks (`remote_single`, `main.py:1730`)
The expensive path. Builds a category-shaped request (`build_request`, `main.py:1351`),
calls `FireworksClient.chat`, and on empty/length issues applies:
- **alt-model fallback** (`alt_model_for`) to another allowed model;
- **length retry** for summarization.

### Batching — NER (`remote_batch`, `main.py:1760`)
NER tasks are batched 8-per-call (`BATCH_SIZE`, `main.py:1288`) against MiniMax to
amortize its fixed serving-template token cost across several short tasks.

## 7. Fireworks client (`FireworksClient`, `main.py:1443`)

- **Single egress point**: every call uses `self.base_url` derived from
  `FIREWORKS_BASE_URL`. No hardcoded hosts.
- **Model restriction**: `resolve_model` (`main.py:1457`) returns only ids present in
  `ALLOWED_MODELS`; unreachable models (HTTP 404) are cached and skipped.
- **Resilience**: param-variant fallback (drops `reasoning_effort` on 400/422),
  multiple URL-path probes, retryable-status handling, per-call timeout.
- **CoT suppression**: `reasoning_effort="none"` by default so thinking models
  (Kimi-K2, MiniMax, …) emit the answer directly; `<think>` blocks are stripped from
  returned content.

## 8. Answer hygiene (`clean_answer`, `main.py:681`)

Strips chain-of-thought leakage (`_strip_cot`, `_strip_reasoning_blocks`), code fences,
intros, trailing periods, outer quotes; normalizes sentiment labels; extracts final
numeric answers. This protects both correctness and the token/format contract.

## 9. Token accounting

Only `remote_single` / `remote_batch` add to `total_tokens`
(`prompt_tokens + completion_tokens`). Local GGUF and all Tier-0 shortcuts add **0**.
This is the entire lever: push as much volume as possible onto the free tiers while
keeping accuracy above the gate.

## 10. Bulletproof output (`write_results`, `main.py:1909`)

Writes results in task order with a 3-attempt fallback (UTF-8 → ASCII-escaped →
empty-placeholder list), guaranteeing valid JSON and a complete `task_id` set.

## 11. Data flow summary

```
run()
 ├─ load_settings()                # harness env
 ├─ Run()                           # load GGUF (lazy), build FireworksClient
 ├─ read /input/tasks.json
 ├─ for each task: _route_category → try solve_zero_token (free)
 │     • hit  -> record (0 tok)
 │     • miss -> route to NER-batch group or single queue
 ├─ flush NER batches (remote, batched, MiniMax)
 ├─ ThreadPoolExecutor over singles: solve_task()
 │     zero-token → local GGUF (0 tok) → remote (tokens) [+ alt/length fallbacks]
 │     apply enforce_limit() if a word/sentence cap was requested
 ├─ assemble in task order, sum tokens, write_results(), exit 0
```

## 12. Why this is competition-compliant

- **One egress**: all remote inference via `FIREWORKS_BASE_URL`.
- **Allowed models only**: `resolve_model` filters `ALLOWED_MODELS`; no other hosts.
- **Local = 0 tokens**: GGUF served in-process; the rules explicitly license this.
- **No baked secrets**: `.env` is not copied; only `main.py`/`local_llm.py`/`requirements.txt`.
- **Contract-safe**: always writes `results.json`, always exits 0; `linux/amd64`.
