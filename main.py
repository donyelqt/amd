"""
AMD Developer Hackathon: ACT II - Track 1
Hybrid Token-Efficient Routing Agent

Local-first, Fireworks fallback. Local inference costs 0 tokens.
All remote calls route through FIREWORKS_BASE_URL using only ALLOWED_MODELS.
"""

import json
import os
import re
import sys
import traceback
from dotenv import load_dotenv

load_dotenv(override=True)

INPUT_PATH = os.environ.get("INPUT_PATH", "/input/tasks.json")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "/output/results.json")

# Confirmed ALLOWED_MODELS from Discord launch-day announcement.
# Ordered cheapest → most expensive (by architecture, verified by name):
#   gemma-4-26b-a4b-it   MoE 26B/4B active  — cheapest remote tier
#   gemma-4-31b-it-nvfp4 Dense 31B 4-bit     — cheap-mid
#   minimax-m3            Dense ~30B           — general purpose
#   gemma-4-31b-it        Dense 31B full      — mid, best accuracy
#   kimi-k2p7-code        Code specialist      — expensive, code only
COST_ORDER = [
    "accounts/fireworks/models/minimax-m3",        # confirmed working, cheap general
    "accounts/fireworks/models/kimi-k2p7-code",    # confirmed working, code specialist
]

# Categories where local inference is safe and high-confidence.
# Sentiment + factual + simple summarisation are reliably handled by a 2B-3B model.
LOCAL_SAFE = {"sentiment", "factual", "summarisation"}

# Categories where local inference is risky — escalate to Fireworks.
LOCAL_RISKY = {"math", "logic", "debugging", "codegen", "ner"}


def load_env() -> dict:
    api_key = os.environ.get("FIREWORKS_API_KEY", "")
    base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
    allowed_raw = os.environ.get("ALLOWED_MODELS", "")
    allowed = [m.strip() for m in allowed_raw.split(",") if m.strip()]
    # Preserve launch-day ordering but also build a cost-sorted list.
    cost_sorted = [m for m in COST_ORDER if m in allowed] + [m for m in allowed if m not in COST_ORDER]
    local_endpoint = os.environ.get("LOCAL_MODEL_ENDPOINT", "")
    local_model_name = os.environ.get("LOCAL_MODEL_NAME", "")
    return {
        "api_key": api_key,
        "base_url": base_url,
        "allowed": allowed,
        "cost_sorted": cost_sorted,
        "local_endpoint": local_endpoint,
        "local_model_name": local_model_name,
    }


# ---------------------------------------------------------------------------
# Category classifier — rule-based, zero-token cost
# ---------------------------------------------------------------------------

def classify_category(prompt: str) -> str:
    """
    Rule-based router. Zero-token cost.
    Returns one of: factual, math, sentiment, summarisation, ner, debugging, logic, codegen
    Order matters: specific structural signals (codegen, debugging, ner) must come
    BEFORE broad word-matches (sentiment) to avoid false positives like "non-negative".
    """
    p = prompt.lower()

    # Strong structural signals — check first
    if re.search(r"\b(write.*function|implement|generate.*code|code that|def\s+\w+|class\s+\w+)\b", p):
        return "codegen"
    if re.search(r"\b(debug|bug|fix.*code|corrected implementation|what is wrong|error in|fix this)\b", p):
        return "debugging"
    if re.search(r"\b(named entity|ner|extract.*entit|person.*org.*location|label.*entit)\b", p):
        return "ner"
    if re.search(r"\b(solve|puzzle|constraint|all conditions|deduce|if.*then|must be|tallest|shortest|older|younger)\b", p):
        return "logic"
    if re.search(r"\b(calculate|how many|percent|sum of|multiply|arithmetic|word problem|\d+\s*[\+\-\*\/]\s*\d+|\d+%)", p):
        return "math"
    if re.search(r"\b(summari[sz]e|condense|in one sentence|briefly describe|tl;dr|key points)\b", p):
        return "summarisation"
    # Sentiment last — the word "negative" appears inside "non-negative", etc.
    # Use word-boundary + require sentiment-specific keywords to avoid false positives.
    if re.search(r"\b(?:sentiment|positive|negative|neutral)\b", p):
        # Guard: don't classify a math/code task that happens to say "non-negative"
        if not re.search(r"\b(?:non[-_]?negative|code|function|def |implement|debug|bug)\b", p):
            return "sentiment"
    return "factual"


# ---------------------------------------------------------------------------
# Model picker — local first (0 tokens), then cheapest adequate Fireworks
# ---------------------------------------------------------------------------

def pick_model(category: str, env: dict) -> tuple[str, str]:
    """
    Returns (model_id, source) where source is "local" or "fireworks".
    LOCAL FIRST: if a local model is configured AND the category is LOCAL_SAFE,
    return the local model (0 tokens).
    Otherwise return the cheapest adequate Fireworks model.
    """
    local_endpoint = env.get("local_endpoint", "")
    local_name = env.get("local_model_name", "")

    # Try local first for safe categories
    if local_endpoint and local_name and category in LOCAL_SAFE:
        return local_name, "local"

    # Fireworks fallback — pick cheapest model that covers this category
    cost_sorted = env.get("cost_sorted", [])
    if not cost_sorted:
        return "", "fireworks"

    # Category → model preference (first match in cost_sorted wins)
    preferred_order = _model_preference(category)
    for pref in preferred_order:
        if pref in cost_sorted:
            return pref, "fireworks"

    # Fallback: cheapest available
    return cost_sorted[0], "fireworks"


def _model_preference(category: str) -> list[str]:
    """Return model IDs in preference order for this category (cheapest first)."""
    table = {
        # MoE handles most things cheaply; code tasks go to code specialist
        "codegen":       ["kimi-k2p7-code"],
        "debugging":     ["kimi-k2p7-code"],
        # Math and logic need stronger reasoning — avoid the MoE if possible
        "math":          ["minimax-m3", "gemma-4-31b-it-nvfp4", "gemma-4-31b-it"],
        "logic":         ["gemma-4-31b-it", "minimax-m3", "gemma-4-31b-it-nvfp4"],
        # NER benefits from a model that handles JSON well
        "ner":           ["gemma-4-31b-it-nvfp4", "minimax-m3", "gemma-4-26b-a4b-it"],
        # General: start with the MoE, escalate only if needed
        "factual":       ["gemma-4-26b-a4b-it", "minimax-m3", "gemma-4-31b-it-nvfp4"],
        "summarisation": ["gemma-4-26b-a4b-it", "minimax-m3", "gemma-4-31b-it-nvfp4"],
        "sentiment":     ["gemma-4-26b-a4b-it", "minimax-m3"],
    }
    return table.get(category, ["gemma-4-26b-a4b-it", "minimax-m3", "gemma-4-31b-it-nvfp4"])


# ---------------------------------------------------------------------------
# Per-category prompt templates — minimise output tokens without losing accuracy
# ---------------------------------------------------------------------------

def build_prompt(category: str, user_prompt: str) -> str:
    templates = {
        "sentiment": (
            "Classify the sentiment of the following text as exactly one of: "
            "positive, negative, or neutral. "
            "Respond with ONLY the label, no explanation.\n\n"
            "Text: {prompt}"
        ),
        "summarisation": (
            "Summarise the following text in exactly one sentence. "
            "Do not add any preamble or explanation.\n\n"
            "Text: {prompt}"
        ),
        "ner": (
            "Extract named entities from the following text. "
            "Respond with ONLY a JSON object mapping entity type to a list of values. "
            "Types: person, org, location, date. "
            "If none found, return empty JSON. "
            "No markdown, no explanation.\n\n"
            "Text: {prompt}"
        ),
        "math": (
            "Solve the following math problem. "
            "Show your reasoning in 1-2 lines, then give ONLY the final numeric answer on its own line.\n\n"
            "Problem: {prompt}"
        ),
        "logic": (
            "Solve the following logic puzzle. "
            "Reason briefly (1-2 sentences), then state ONLY the final answer.\n\n"
            "Puzzle: {prompt}"
        ),
        "codegen": (
            "Write a correct, minimal implementation for the following specification. "
            "Return ONLY the code, no explanation.\n\n"
            "Spec: {prompt}"
        ),
        "debugging": (
            "Identify the bug in the following code and provide the corrected implementation. "
            "Return ONLY the corrected code with a one-line comment explaining the fix.\n\n"
            "Code: {prompt}"
        ),
        "factual": (
            "Answer the following question concisely. "
            "One paragraph maximum.\n\n"
            "Question: {prompt}"
        ),
    }
    tmpl = templates.get(category, "{prompt}")
    return tmpl.format(prompt=user_prompt)


# ---------------------------------------------------------------------------
# Fireworks client
# ---------------------------------------------------------------------------

def call_fireworks(model: str, prompt: str, env: dict, max_tokens: int = 512) -> tuple[str, int]:
    try:
        from openai import OpenAI
    except ImportError:
        print(
            "ERROR: openai is not installed. "
            "Run: pip install openai",
            file=sys.stderr,
        )
        return "", 0
    client = OpenAI(api_key=env["api_key"], base_url=env["base_url"])
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a precise assistant. Answer exactly as instructed, nothing more."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    answer = (resp.choices[0].message.content or "").strip()
    tokens = (resp.usage.total_tokens if resp.usage else 0) if hasattr(resp, "usage") and resp.usage else 0
    return answer, tokens


# ---------------------------------------------------------------------------
# Local model client — Ollama-compatible endpoint (configurable)
# ---------------------------------------------------------------------------

def call_local(model: str, prompt: str, env: dict, max_tokens: int = 512) -> tuple[str, int]:
    """
    Call a local model via an OpenAI-compatible endpoint (Ollama, vLLM, etc.).
    Configure via env: LOCAL_MODEL_ENDPOINT, LOCAL_MODEL_NAME.
    Tokens are 0 for scoring purposes. Returns (answer, 0).
    """
    endpoint = env.get("local_endpoint", "")
    local_name = env.get("local_model_name", model)
    if not endpoint:
        return "", 0
    try:
        import urllib.request
        url = f"{endpoint.rstrip('/')}/chat/completions"
        payload = json.dumps({
            "model": local_name,
            "messages": [
                {"role": "system", "content": "You are a precise assistant. Answer exactly as instructed, nothing more."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }).encode()
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
            answer = (data.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()
            return answer, 0
    except Exception as e:
        print(f"[local-model] {e}", file=sys.stderr)
        return "", 0


# ---------------------------------------------------------------------------
# Mock — no key, no local endpoint
# ---------------------------------------------------------------------------

def call_mock(model: str, prompt: str, env: dict, max_tokens: int = 512) -> str:
    return f"[MOCK-{model}] {prompt[:80]}..."


# ---------------------------------------------------------------------------
# Process one task
# ---------------------------------------------------------------------------

def process_task(task: dict, env: dict) -> tuple[str, int, str]:
    prompt = task.get("prompt", "")
    category = classify_category(prompt)
    model, source = pick_model(category, env)
    styled = build_prompt(category, prompt)

    max_tokens = {
        "ner": 256, "sentiment": 64, "summarisation": 128,
        "factual": 256, "math": 128, "logic": 256,
        "debugging": 1024, "codegen": 1024,
    }.get(category, 512)

    if source == "local":
        answer, tokens = call_local(model, styled, env, max_tokens=max_tokens)
        if not answer and env.get("api_key") and env.get("allowed"):
            fireworks_model, _ = pick_model(category, env)
            if fireworks_model:
                print(f"[{task.get('task_id','')}] local miss → escalating to {fireworks_model}", file=sys.stderr)
                answer, tokens = call_fireworks(fireworks_model, styled, env, max_tokens=max_tokens)
        return answer, tokens, source

    if source == "fireworks":
        if env.get("api_key"):
            answer, tokens = call_fireworks(model, styled, env, max_tokens=max_tokens)
            return answer, tokens, source
        return call_mock(model, styled, env, max_tokens=max_tokens), 0, "mock"

    return "", 0, "none"


def main() -> int:
    env = load_env()

    if env.get("api_key") and not env.get("allowed"):
        print(
            "ERROR: FIREWORKS_API_KEY is set but ALLOWED_MODELS is empty. Cannot route any task.",
            file=sys.stderr,
        )
        return 2

    if not os.path.exists(INPUT_PATH):
        print(f"ERROR: input not found at {INPUT_PATH}", file=sys.stderr)
        return 1

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    results = []
    debug = []
    total_tokens = 0
    for i, task in enumerate(tasks):
        task_id = task.get("task_id") or f"t{i}"
        try:
            answer, tokens, src = process_task(task, env)
        except Exception:
            traceback.print_exc(file=sys.stderr)
            answer, tokens, src = "", 0, "error"
        total_tokens += tokens
    results.append({"task_id": task_id, "answer": answer})

    debug.append(
        {
            "task_id": task_id,
            "category": classify_category(task.get("prompt", "")),
            "source": src,
            "tokens": tokens,
        }
    )
        print(f"[{task_id}] done ({src}) tokens={tokens}", file=sys.stderr)

    print(f"Total Fireworks tokens: {total_tokens}", file=sys.stderr)

    out_dir = os.path.dirname(OUTPUT_PATH)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    debug_path = os.path.join(out_dir or ".", "debug.json")
    with open(debug_path, "w", encoding="utf-8") as f:
        json.dump({"tasks": debug, "total_tokens": total_tokens}, f, ensure_ascii=False, indent=2)

    print(f"Debug telemetry written to {debug_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
