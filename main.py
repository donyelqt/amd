"""
AMD Developer Hackathon: ACT II - Track 1
Hybrid Token-Efficient Routing Agent

Local-first via llama-cpp-python (GGUF, 0 tokens for scoring),
Fireworks fallback for risky categories. All inference logs to
/output/results.json (harness spec) and /output/debug.json (your telemetry).
"""

import json
import os
import re
import sys
import traceback
from dotenv import load_dotenv

load_dotenv()

INPUT_PATH = os.environ.get("INPUT_PATH", "/input/tasks.json")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "/output/results.json")

# ---------------------------------------------------------------------------
# Config — cheaper / more expensive Fireworks models (ordered for escalation)
# ---------------------------------------------------------------------------
COST_ORDER = [
 "accounts/fireworks/models/minimax-m3",
  "accounts/fireworks/models/gemma-4-31b-it-nvfp4",
    "accounts/fireworks/models/gemma-4-31b-it",
    "accounts/fireworks/models/gemma-4-26b-a4b-it",
    "accounts/fireworks/models/kimi-k2p7-code",
]

LOCAL_SAFE = {"sentiment", "factual", "summarisation"}
LOCAL_RISKY = {"math", "logic", "debugging", "codegen", "ner"}

# ---------------------------------------------------------------------------
# llm singleton
# ---------------------------------------------------------------------------
_LLM = None


def _get_llm(env: dict):
    global _LLM
    if _LLM is not None:
        return _LLM
    try:
        from llama_cpp import Llama
    except ImportError:
        print("[local] llama-cpp-python not installed", file=sys.stderr)
        return None
    model_path = env.get("LOCAL_MODEL_PATH", "/app/models/qwen2.5-0.5b-instruct-q4_k_m.gguf")
    if os.path.exists(model_path) and os.path.getsize(model_path) > 1_000_000:
        try:
            _LLM = Llama(
                model_path=model_path,
                n_ctx=int(env.get("LOCAL_MODEL_N_CTX", "2048")),
                n_threads=int(env.get("LOCAL_MODEL_N_THREADS", "4")),
                verbose=False,
            )
            print(f"[local] loaded {model_path}", file=sys.stderr)
            return _LLM
        except Exception as exc:
            print(f"[local] load error: {exc}", file=sys.stderr)
    repo_id = env.get("LOCAL_MODEL_REPO", "Qwen/Qwen2.5-0.5B-Instruct-GGUF")
    filename = env.get("LOCAL_MODEL_FILE", "qwen2_5-0.5b-instruct-q4_k_m.gguf")
    hf_token = os.environ.get("HF_TOKEN") or env.get("HF_TOKEN", "")
    try:
        print(f"[local] downloading {repo_id}:{filename} ...", file=sys.stderr)
        _LLM = Llama.from_pretrained(
            repo_id=repo_id,
            filename=filename,
            n_ctx=int(env.get("LOCAL_MODEL_N_CTX", "2048")),
            n_threads=int(env.get("LOCAL_MODEL_N_THREADS", "4")),
            verbose=False,
            hf_api_token=hf_token or None,
        )
        print("[local] model ready", file=sys.stderr)
        return _LLM
    except Exception as exc:
        print(f"[local] download error: {exc}", file=sys.stderr)
        return None


def _local_chat(env: dict, messages: list, max_tokens: int = 512) -> tuple[str, int]:
    llm = _get_llm(env)
    if llm is None:
        return "", 0
    try:
        resp = llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0,
            stop=["\n\n", "User:", "Human:"],
        )
        text = (resp.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()
        tokens = 0  # local tokens count as zero per hackathon rules
        return text, tokens
    except Exception as exc:
        print(f"[local] inference error: {exc}", file=sys.stderr)
        return "", 0

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def load_env() -> dict:
    api_key = os.environ.get("FIREWORKS_API_KEY", "")
    base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
    allowed_raw = os.environ.get("ALLOWED_MODELS", "")
    allowed = [m.strip() for m in allowed_raw.split(",") if m.strip()]
    cost_sorted = [m for m in COST_ORDER if m in allowed] + [m for m in allowed if m not in COST_ORDER]
    return {
        "api_key": api_key,
        "base_url": base_url,
        "allowed": allowed,
        "cost_sorted": cost_sorted,
        "local_model_endpoint": os.environ.get("LOCAL_MODEL_ENDPOINT", ""),
        "local_model_name": os.environ.get("LOCAL_MODEL_NAME", "qwen2.5:0.5b"),
        "local_model_path": os.environ.get("LOCAL_MODEL_PATH", "/app/models/qwen2.5-0.5b-instruct-q4_k_m.gguf"),
        "local_model_n_ctx": os.environ.get("LOCAL_MODEL_N_CTX", "2048"),
        "local_model_n_threads": os.environ.get("LOCAL_MODEL_N_THREADS", "4"),
    }

# ---------------------------------------------------------------------------
# Category classifier — rule-based, zero tokens
# ---------------------------------------------------------------------------

def classify_category(prompt: str) -> str:
    p = prompt.lower()
    if re.search(r"\b(write.*function|implement|generate.*code|def\s+\w+|class\s+\w+)\b", p):
        return "codegen"
    if re.search(r"\b(debug|bug|fix.*code|corrected implementation|what is wrong|error in|fix this)\b", p):
        return "debugging"
    if re.search(r"\b(named entity|ner|extract.*entit|person.*org.*location|label.*entit)\b", p):
        return "ner"
    if re.search(r"\b(solve|puzzle|constraint|all conditions|deduce|if.*then|must be|tallest|shortest)\b", p):
        return "logic"
    if re.search(r"\b(calculate|how many|percent|sum of|multiply|\d+\s*[\+\-\*\/]\s*\d+|\d+%)", p):
        return "math"
    if re.search(r"\b(summari[sz]e|condense|in one sentence|tl;dr|key points)\b", p):
        return "summarisation"
    if re.search(r"\b(sentiment|positive|negative|neutral)\b", p):
        if not re.search(r"\b(non[-_]?negative|code|function|def |implement|debug|bug)\b", p):
            return "sentiment"
    return "factual"

# ---------------------------------------------------------------------------
# Model picker — local first (0 tokens), then cheapest Fireworks
# ---------------------------------------------------------------------------

def pick_model(category: str, env: dict) -> tuple[str, str]:
    if category in LOCAL_SAFE:
        if env.get("local_model_endpoint"):
            return env["local_model_name"], "local"
        llm = _get_llm(env)
        if llm is not None:
            return "local", "local"
    cost_sorted = env.get("cost_sorted", [])
    if not cost_sorted:
        return "", "fireworks"
    preferred = _model_preference(category)
    for pref in preferred:
        if pref in cost_sorted:
            return pref, "fireworks"
    return cost_sorted[0], "fireworks"


def _model_preference(category: str) -> list[str]:
    table = {
        "codegen": ["accounts/fireworks/models/kimi-k2p7-code"],
        "debugging": ["accounts/fireworks/models/kimi-k2p7-code"],
        "math": ["accounts/fireworks/models/minimax-m3"],
        "logic": ["accounts/fireworks/models/minimax-m3"],
        "ner": ["accounts/fireworks/models/minimax-m3"],
        "factual": ["accounts/fireworks/models/minimax-m3"],
        "summarisation": ["accounts/fireworks/models/minimax-m3"],
        "sentiment": ["accounts/fireworks/models/minimax-m3"],
    }
    return table.get(category, ["accounts/fireworks/models/minimax-m3"])

# ---------------------------------------------------------------------------
# Per-category prompts — minimise output tokens
# ---------------------------------------------------------------------------

def build_prompt(category: str, user_prompt: str) -> str:
    templates = {
        "sentiment": (
            "Classify sentiment as exactly one of: positive, negative, or neutral. "
            "Respond with ONLY the label.\n\nText: {prompt}"
        ),
        "summarisation": (
            "Summarise in exactly one sentence. No preamble.\n\nText: {prompt}"
        ),
        "ner": (
            "Extract named entities as JSON: {{'person':[],'org':[],'location':[],'date':[]}}. "
            "No markdown.\n\nText: {prompt}"
        ),
        "math": (
            "Solve. Show 1-2 lines of reasoning, then ONLY the final numeric answer.\n\nProblem: {prompt}"
        ),
        "logic": (
            "Solve briefly (1-2 sentences), then state ONLY the final answer.\n\nPuzzle: {prompt}"
        ),
        "codegen": (
            "Return ONLY the code, no explanation.\n\nSpec: {prompt}"
        ),
        "debugging": (
            "Return ONLY corrected code with a one-line fix comment.\n\nCode: {prompt}"
        ),
        "factual": (
            "Answer concisely. One paragraph maximum.\n\nQuestion: {prompt}"
        ),
    }
    return templates.get(category, "{prompt}").format(prompt=user_prompt)

# ---------------------------------------------------------------------------
# Fireworks client
# ---------------------------------------------------------------------------

def call_fireworks(model: str, prompt: str, env: dict, max_tokens: int = 512) -> tuple[str, int]:
    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: openai not installed", file=sys.stderr)
        return "", 0
    client = OpenAI(api_key=env["api_key"], base_url=env["base_url"])
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a precise assistant. Answer exactly as instructed."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        answer = (resp.choices[0].message.content or "").strip()
        tokens = resp.usage.total_tokens if resp.usage else 0
        return answer, tokens
    except Exception as exc:
        print(f"[fireworks] {model} error: {exc}", file=sys.stderr)
        return "", 0

# ---------------------------------------------------------------------------
# Local client — llama-cpp-python, 0-token inference
# ---------------------------------------------------------------------------

def call_local(model: str, prompt: str, env: dict, max_tokens: int = 512) -> tuple[str, int]:
    endpoint = env.get("local_model_endpoint", "")
    name = env.get("local_model_name", "")
    if endpoint:
        try:
            import urllib.request
            import json as _json
            url = f"{endpoint.rstrip('/')}/chat/completions"
            body = _json.dumps({
                "model": name,
                "messages": [
                    {"role": "system", "content": "You are a precise assistant. Answer exactly as instructed."},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": 0.0,
            }).encode()
            req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = _json.loads(resp.read())
                text = (data.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()
                return text, 0
        except Exception as exc:
            print(f"[local] ollama {name} error: {exc}", file=sys.stderr)
            return "", 0
    messages = [
        {"role": "system", "content": "You are a precise assistant. Answer exactly as instructed."},
        {"role": "user", "content": prompt},
    ]
    return _local_chat(env, messages, max_tokens=max_tokens)

# ---------------------------------------------------------------------------
# Mock fallback (no key, no local model)
# ---------------------------------------------------------------------------

def call_mock(model: str, prompt: str, env: dict, max_tokens: int = 512) -> str:
    return f"[MOCK-{model}] {prompt[:80]}..."

# ---------------------------------------------------------------------------
# Task processing
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
            fw_env = {k: v for k, v in env.items() if k not in ("local_model_path", "local_model_n_ctx", "local_model_n_threads")}
            fw_model, _ = pick_model(category, fw_env)
            if fw_model:
                print(f"[{task.get('task_id','')}] local miss → {fw_model}", file=sys.stderr)
                answer, tokens = call_fireworks(fw_model, styled, env, max_tokens=max_tokens)
        if not answer:
            print(f"[{task.get('task_id','')}] empty answer", file=sys.stderr)
        return answer, tokens, source

    if source == "fireworks":
        if env.get("api_key"):
            answer, tokens = call_fireworks(model, styled, env, max_tokens=max_tokens)
            if not answer:
                llm = _get_llm(env)
                if llm is not None:
                    print(f"[{task.get('task_id','')}] fireworks miss → local", file=sys.stderr)
                    answer, _ = call_local("local", styled, env, max_tokens=max_tokens)
                    tokens = 0
            return answer, tokens, source
        print(f"[{task.get('task_id','')}] no API key, mock fallback", file=sys.stderr)
        return call_mock(model, styled, env), 0, "mock"

    print(f"[{task.get('task_id','')}] no model", file=sys.stderr)
    return "", 0, "none"

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    try:
        env = load_env()
    except Exception as exc:
        print(f"ERROR: env load failed: {exc}", file=sys.stderr)
        return 2

    if env.get("api_key") and not env.get("allowed"):
        print("ERROR: FIREWORKS_API_KEY set but ALLOWED_MODELS empty", file=sys.stderr)
        return 2

    if not os.path.exists(INPUT_PATH):
        print(f"ERROR: input not found at {INPUT_PATH}", file=sys.stderr)
        return 1

    try:
        with open(INPUT_PATH, "r", encoding="utf-8") as f:
            tasks = json.load(f)
    except Exception as exc:
        print(f"ERROR: failed to read input: {exc}", file=sys.stderr)
        return 1

    results = []
    debug = []
    total_tokens = 0

    try:
        if any(classify_category(t.get("prompt", "")) in LOCAL_SAFE for t in tasks):
            _get_llm(env)
    except Exception:
        pass

    for i, task in enumerate(tasks):
        task_id = task.get("task_id") or f"t{i}"
        try:
            answer, tokens, src = process_task(task, env)
        except Exception:
            traceback.print_exc(file=sys.stderr)
            answer, tokens, src = "", 0, "error"
        total_tokens += tokens
        results.append({"task_id": task_id, "answer": answer})
        debug.append({
            "task_id": task_id,
            "category": classify_category(task.get("prompt", "")),
            "source": src,
            "tokens": tokens,
        })
        print(f"[{task_id}] done ({src}) tokens={tokens}", file=sys.stderr)

    print(f"Total Fireworks tokens: {total_tokens}", file=sys.stderr)

    try:
        out_dir = os.path.dirname(OUTPUT_PATH)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        debug_path = os.path.join(out_dir or ".", "debug.json")
        with open(debug_path, "w", encoding="utf-8") as f:
            json.dump({"tasks": debug, "total_tokens": total_tokens}, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        print(f"ERROR: failed to write output: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
