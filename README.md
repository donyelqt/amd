# Hybrid Token-Efficient Routing Agent — Track 1
## AMD Developer Hackathon: ACT II

Local-first, Fireworks fallback. Safe categories run on a local model inside the
container for **0 tokens**. Hard categories escalate to the cheapest adequate
Fireworks model from `ALLOWED_MODELS`. Routing intelligence wins, not raw compute.

## Files

| File | Purpose |
|---|---|
| `main.py` | Entry point: read `/input/tasks.json` → classify → local first → Fireworks fallback → write `/output/results.json` |
| `Dockerfile` | `python:3.11-slim`, zero bloat, no web server |
| `.env.example` | Local dev env vars. Copy to `.env`; **never commit a real key** |
| `sample_tasks.json` | One task per category for local testing |
| `.gitignore` | Ignores `.env`, `output/`, `__pycache__/` |

## Not in this repo (and never will be)

No FastAPI, no React, no Next.js, no web server, no HTTP endpoint.
Track 1 is a **batch container**: harness mounts a JSON file, runs `python main.py`,
reads `/output/results.json`, scores it. That's the whole interaction.

## Environment (harness-injected at eval time)

| Variable | Description |
|---|---|
| `FIREWORKS_API_KEY` | Provided by the harness — use this, **not your own** |
| `FIREWORKS_BASE_URL` | ALL Fireworks calls must route through this URL |
| `ALLOWED_MODELS` | Comma-separated list of permitted Fireworks model IDs |
| `LOCAL_MODEL_ENDPOINT` | *(optional)* OpenAI-compatible local endpoint, e.g. `http://localhost:11434/v1` |
| `LOCAL_MODEL_NAME` | *(optional)* Model name at the local endpoint, e.g. `gemma:2b` |

> Harness explicitly says: do not bundle `.env` in the image.
> Only `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, `ALLOWED_MODELS` are injected
> at eval time. `LOCAL_MODEL_*` vars are for your dev setup only.

## How to run locally

```powershell
# Windows
$env:INPUT_PATH  = "sample_tasks.json"
$env:OUTPUT_PATH = "output/results.json"
python main.py
type output/results.json
```

```bash
# Linux / macOS
INPUT_PATH=sample_tasks.json OUTPUT_PATH=output/results.json python main.py
cat output/results.json
```

With local model (Ollama running on localhost:11434):

```bash
cp .env.example .env
# Edit .env: uncomment LOCAL_MODEL_ENDPOINT and LOCAL_MODEL_NAME
INPUT_PATH=sample_tasks.json OUTPUT_PATH=output/results.json python main.py
```

## How to build the Docker image

> **Image size budget: 10 GB compressed.** Keep it lean.

```bash
docker build -t track1-agent .
```

Mock-mode test (no GPU, no key):

```bash
docker run --rm `
  -v "${PWD}/sample_tasks.json:/input/tasks.json:ro" `
  -v "${PWD}/output:/output" `
  track1-agent
```

Real-fireworks test (requires `pip install openai` on your host for the test run):

```bash
docker run --rm `
  -e FIREWORKS_API_KEY=$FIREWORKS_API_KEY `
  -e FIREWORKS_BASE_URL=$FIREWORKS_BASE_URL `
  -e ALLOWED_MODELS=$ALLOWED_MODELS `
  -v "${PWD}/sample_tasks.json:/input/tasks.json:ro" `
  -v "${PWD}/output:/output" `
  track1-agent
```

> The harness **always** injects `FIREWORKS_API_KEY` at eval time.
> Your container must read it from the environment; never bake it into the image.

## What gets scored

1. **Accuracy gate first** — LLM-Judge checks each answer against expected intent.
   Submissions below threshold are excluded from the leaderboard.
2. **Token efficiency second** — among passing submissions, rank = fewest Fireworks
   tokens recorded by the judging proxy.
   - Local inference (via `LOCAL_MODEL_*`) = **0 tokens** ✅
   - Fireworks calls through `FIREWORKS_BASE_URL` = **every token counts**

## Winning strategy (verified)

| Category | First try | Escalates to Fireworks if… |
|---|---|---|
| Sentiment | Local 2B–3B | Low confidence | `gemma-4-26b-a4b-it` |
| Factual | Local 2B–3B | Low confidence | `gemma-4-26b-a4b-it` |
| Summarisation | Local 2B–3B | Low confidence | `gemma-4-26b-a4b-it` |
| NER | Local 2B–3B | JSON parse fails | `gemma-4-31b-it-nvfp4` |
| Math | — | Always | `minimax-m3` or `gemma-4-31b-it` |
| Logic | — | Always | `gemma-4-31b-it` |
| Code debugging | — | Always | `kimi-k2p7-code` |
| Code generation | — | Always | `kimi-k2p7-code` |

## Constraints honoured

- [x] Exit code 0 on success, non-zero on failure
- [x] Maximum runtime: 10 minutes
- [x] Reads `/input/tasks.json`, writes `/output/results.json`
- [x] All env vars read from `os.environ` only — nothing hardcoded in the image
- [x] Per-task exception trapping — one failure doesn't kill the whole run
- [x] Valid JSON output always flushed before exit
- [x] Image compressed size well under 10 GB cap (no web server, no CUDA)

## TODO before submit

- [ ] Test with real Fireworks key + `ALLOWED_MODELS` — measure `usage.total_tokens`
- [ ] Test local model on your RTX 3050 (4 GB VRAM) — likely `gemma:2b` or `gemma:3b` 4-bit
- [ ] Tune per-category prompts to shave output tokens
- [ ] Validate accuracy gate on held-out samples per category
- [ ] Push image to GHCR / Docker Hub, submit on lablab.ai
