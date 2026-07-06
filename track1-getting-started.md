# Track 1 — Hybrid Token-Efficient Routing Agent
## Getting Started Guide

---

## What Track 1 Actually Is

You are building an **autonomous routing agent** that receives a task, decides whether to answer it with a **local model** or a **remote model (Fireworks AI API)**, and must complete the task while minimizing **remote token usage** (because local tokens count as `$0$` in the score) **without dropping below accuracy threshold**.

The scoring combines:
- **Token count** (only remote API tokens matter)
- **Output accuracy** (judged either by a leaderboard eval or a provided validation set)

---

## Step 1 — Confirm Enrollment & API Access (Do This First)

1. **AMD AI Developer Program (ADP)**  
   - Sign up at the AMD ADP link from your hackathon page.  
   - If you registered before **July 2nd**, you should already have credits. If not, allocations start July 7.

2. **Fireworks AI API**  
   - You should receive API credits tied to your ADP account.  
   - Get your API key from the Fireworks / AMD dashboard.  
   - Install the SDK: `pip install fireworks-ai` (or use `openai` if they expose an OpenAI-compatible endpoint, which Fireworks does).

3. **Test that your API key works**  
   - Make a single remote model call (e.g., to one of the models they list) to confirm connectivity and credit availability.

---

## Step 2 — Understand the Evaluation Environment

From the rules:
- Final scoring runs on a **standardized environment** (hardware + OS + model constraints are fixed).
- **Local models must be sized to run inside that environment.**  
  If you pick a 70B parameter model for local but the eval box only has 16GB VRAM, you fail.

**Questions to resolve now:**
- Does the hackathon platform publish the eval hardware specs? (Check the hackathon dashboard, Discord, or Getting Started Guide.)
- Are there example evaluation tasks or sample inputs available?
- How do you submit? (CLI upload? API endpoint? GitHub link?)

If there is a **local evaluation harness** or **judge API** provided, locate it now. Many hackathons give a Python script or Docker image you run locally to self-evaluate before submission. Run that script once even with a dummy answer to confirm it executes.

---

## Step 3 — Choose Your Local Model Strategy

Because local tokens are free, your router should default to local whenever feasible. The real design problem is:

| Model Location | Token Cost to Score | Latency Risk | Accuracy Risk |
|---|---|---|---|
| Local (eval box) | $0$ | Higher on weak hardware | Varies |
| Remote (Fireworks) | Costs apply | Lower | Generally higher |

Before coding, decide:
- **Which local model to use?** Pick something lightweight that fits in the standardized eval environment (e.g., Gemma 2B/7B, Llama 3.2 1B/3B, Phi-3 Mini, Qwen2.5 3B/7B).
- **What remote model for hard tasks?** Identify 1–2 Fireworks models with strong accuracy that you will fall back to.
- **What routing heuristic?** Will you use a simple rule (e.g., token-length threshold, task-type classifier), or a learned router?

---

## Step 4 — Understand the Task Format

According to the schedule, tasks are revealed at kickoff (~12:35 AM PST). When they drop:
- Read the task prompt carefully.
- Check if tasks are **single-turn** or **multi-turn**.
- Check if there is an **accuracy threshold** per task.
- Note whether you are given a **validation set** for local testing.

---

## Step 5 — Set Up a Local Dev Container

Because the final eval is containerized, build your environment in Docker from day one:
- Base image that matches the eval specs (if provided).
- Preinstall your local model weights (or plan to download at runtime).
- Install Fireworks client.
- Add a simple `main.py` that you can run locally to test routing.

---

## Pre-Coding Checklist

- [ ] Registered for AMD AI Developer Program
- [ ] Fireworks API key active and tested
- [ ] Confirmed eval hardware specs / Docker image (if available)
- [ ] Downloaded / pinned a local model compatible with eval specs
- [ ] Located the evaluation harness / submission instructions
- [ ] Read the task specification
- [ ] Set up a local Docker container matching the eval environment
- [ ] Created a minimal script that calls local + remote and returns a completion
