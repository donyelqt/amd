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

## How to build and push the Docker image

```bash
# Build for linux/amd64 (required by judging VM)
docker buildx build --platform linux/amd64 --tag donyelqt/track1-agent:latest --push .

# Tag for GHCR
docker tag donyelqt/track1-agent:latest ghcr.io/donyelqt/amd/track1-agent:latest

# Push to GHCR (make sure package is public at github.com/users/donyelqt/packages)
docker push ghcr.io/donyelqt/amd/track1-agent:latest
```

## How to run for evaluation

```powershell
# Windows PowerShell
docker run --rm `
  -v "${PWD}\input:/input:ro" `
  -v "${PWD}\output:/output" `
  -e FIREWORKS_API_KEY="your-key" `
  -e FIREWORKS_BASE_URL="https://api.fireworks.ai/inference/v1" `
  -e ALLOWED_MODELS="accounts/fireworks/models/minimax-m3" `
  donyelqt/track1-agent:latest
```

```bash
# Linux / macOS
docker run --rm \
  -v "${PWD}/input:/input:ro" \
  -v "${PWD}/output:/output" \
  -e FIREWORKS_API_KEY="your-key" \
  -e FIREWORKS_BASE_URL="https://api.fireworks.ai/inference/v1" \
  -e ALLOWED_MODELS="accounts/fireworks/models/minimax-m3" \
  donyelqt/track1-agent:latest
```

> The harness **always** injects `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, and `ALLOWED_MODELS` at eval time.
> Your container reads them from the environment; never bake them into the image.

> **Note**: `/input/tasks.json` and `/output/` are mounted by the evaluation system. Do not bundle them in the image.
> If running locally without volume mounts, place `tasks.json` in `./input/` or use `sample_tasks.json`.

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

- [x] Test with real Fireworks key + `ALLOWED_MODELS` — measure `usage.total_tokens`
- [x] Test local model inference (Qwen2.5-0.5B GGUF inside container)
- [x] Tune per-category prompts to shave output tokens
- [x] Validate accuracy gate on held-out samples per category
- [x] Push image to GHCR, submit on lablab.ai
