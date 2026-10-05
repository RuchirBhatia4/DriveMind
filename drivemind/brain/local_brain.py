"""On-device brain: Qwen2.5-VL-3B, 4-bit quantized, running on Apple MLX.

Concepts:

* **Vision-Language Model (VLM) architecture.** A ViT image encoder turns the frame into
  patch embeddings, a projector maps them into the LLM's token space, and the LLM attends
  over image tokens and text tokens together. "Seeing" is just more tokens in the context.
* **4-bit weight quantization.** 3B params x 16 bits = 6 GB. At 4 bits (group-wise affine
  quantization: each group of 64 weights shares a scale + bias) that's ~2 GB, with a small
  quality loss. Decoding is memory-bandwidth-bound, so fewer bytes per weight is *directly*
  more tokens/sec. This is the single most important trick for edge LLMs.
* **Prefill vs. decode.** Prefill processes the whole prompt (incl. image tokens) in
  parallel and is compute-bound; decode generates one token at a time and is
  bandwidth-bound. We report both speeds so you can see which one dominates latency.
* **Prompted tool calling + constrained output.** Small local models don't have a native
  tool API, so we describe tools in the prompt and ask for JSON, then parse defensively.
  (Production systems use grammar-constrained decoding to *guarantee* valid JSON.)
* **Route deterministically where you can.** Simple rules (common.py) pick one of four paths:
  visual memory for questions about the past, a describe-only prompt for go/no-go safety
  questions, the tool prompt for car commands, and a plain look-and-answer prompt for
  everything else. Leaving these choices to the 3B model failed in measurable ways
  (eval/brain_qa_eval.py): it used memory for "which car is on my left?", copied example
  answers ("the light is green" at a red light), and said "it's safe to go".
"""

from __future__ import annotations

import json
import re
import threading
import time

from drivemind.brain.common import (
    SYSTEM_PROMPT,
    BrainContext,
    BrainResult,
    command_intent,
    downscale_jpeg,
    ground_light_claims,
    light_sentence,
    memory_intent,
    mentions_light_colour,
    safety_intent,
)
from drivemind.brain.tools import TOOL_NAMES, execute_vehicle_tool, tools_prompt_block
from drivemind.config import settings

JSON_INSTRUCTIONS = """Available tools:
{tools}

Respond with ONLY one JSON object, no other text:
{{"tool_calls": [{{"name": "<tool>", "args": {{...}}}}], "say": "<what to say aloud>"}}
A request can need several tools: include one entry per action.

Examples:
Driver says: "I'm freezing, warm it up and put on some Taylor Swift"
{{"tool_calls": [{{"name": "set_temperature", "args": {{"temperature_f": 74}}}}, {{"name": "play_media", "args": {{"query": "Taylor Swift"}}}}], "say": "Warming it up to 74 and playing Taylor Swift."}}
Driver says: "The windshield is fogging up, and take me home"
{{"tool_calls": [{{"name": "set_defrost", "args": {{"on": true}}}}, {{"name": "navigate", "args": {{"destination": "home"}}}}], "say": "Defroster on, and heading home."}}"""
# Only car commands reach this prompt (see answer()), so it has no visual-question examples:
# the 3B model copied them word for word.

_SAFETY_CLAIM = re.compile(r"\bsafe\b|\byou can (?:go|turn|merge|proceed|pass)\b|\bgo ahead\b|\bit'?s clear\b", re.I)


def _looks_like_json(text: str) -> bool:
    return "{" in text or "tool_calls" in text or '"say"' in text


def _extract_json(text: str) -> dict | None:
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


class LocalBrain:
    name = "local"

    def __init__(self, mlx_lock: threading.Lock) -> None:
        from mlx_vlm import load

        self.lock = mlx_lock  # MLX work (Whisper + VLM) is serialized on the GPU
        self.model, self.processor = load(settings.local_vlm)
        self.config = self.model.config

    def _generate(self, system: str, user: str, image, max_tokens: int) -> tuple[str, dict]:
        from mlx_vlm import apply_chat_template, generate

        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = apply_chat_template(self.processor, self.config, messages, num_images=1 if image else 0)
        with self.lock:
            out = generate(
                self.model, self.processor, prompt, image=[image] if image else None,
                max_tokens=max_tokens, temperature=0.0, verbose=False,
            )
        stats = {
            "prompt_tokens": out.prompt_tokens,
            "generation_tokens": out.generation_tokens,
            "prefill_tps": round(out.prompt_tps, 1),
            "decode_tps": round(out.generation_tps, 1),
            "peak_mem_gb": round(out.peak_memory, 2),
        }
        return out.text, stats

    def _plain_answer(self, query: str, image, note: str = "", context: str = "") -> tuple[str, dict]:
        prompt = (f"{note}{context}Driver says: \"{query}\"\nAnswer in one short spoken sentence, from what you "
                  "actually see in the image, even if the question suggests something else.")
        return self._generate(SYSTEM_PROMPT, prompt, image, max_tokens=80)

    @staticmethod
    def _plain_context(ctx: BrainContext) -> str:
        # Detection counts help with "how many" questions, and traffic-light colours come from the
        # lamp detector (eval/traffic_light_eval.py). Recent turns allow follow-ups.
        hist = " ".join(f"Driver: {u} You: {a}" for u, a in ctx.history[-3:])
        return f"Perception: {ctx.road_summary}\n" + (f"Recent conversation: {hist}\n" if hist else "")

    def _result(self, say: str, calls, retrieved, t0: float, detail: dict) -> BrainResult:
        return BrainResult(text=say.strip(), brain=self.name, tool_calls=calls, retrieved=retrieved,
                           ms=round((time.perf_counter() - t0) * 1000, 1), detail=detail)

    def answer(self, ctx: BrainContext, memory=None) -> BrainResult:
        t0 = time.perf_counter()
        calls: list[dict] = []
        retrieved: list[dict] = []
        q = ctx.query

        # 1. About something already passed: answer from visual memory (RAG over video).
        if memory is not None and memory_intent(q):
            hits = memory.search(q, k=1)
            calls.append({"name": "recall_scene", "args": {"query": q}, "result": f"{len(hits)} frame(s)"})
            if not hits:
                return self._result("I don't have anything in my recent visual memory that matches.", calls, [], t0, {"route": "memory"})
            hit = hits[0]
            retrieved.append({k: hit[k] for k in ("seconds_ago", "similarity")})
            past = downscale_jpeg(hit["jpeg"], settings.vlm_image_width * 2)  # more pixels to read text
            say, gen = self._plain_answer(q, past, note=f"This image is the road camera view from {hit['seconds_ago']:.0f} seconds ago. ")
            return self._result(say, calls, retrieved, t0, {"route": "memory", **gen})

        image = downscale_jpeg(ctx.road_jpeg, settings.vlm_image_width) if ctx.road_jpeg else None

        # 2. Go/no-go questions: describe what's there; the decision stays with the driver.
        if safety_intent(q) and not command_intent(q):
            obs, gen = self._generate(SYSTEM_PROMPT, (
                "In one short sentence, describe what you see ahead that matters for this question. "
                f"Do not say whether it is safe or what the driver should do: \"{q}\""), image, max_tokens=60)
            # Code has the last word: no "it's safe", and the light's colour comes from the lamp detector.
            kept = [s for s in re.split(r"(?<=[.!?])\s+", obs.strip()) if s and not _SAFETY_CLAIM.search(s)
                    and not (ctx.lights is not None and mentions_light_colour(s))]
            lead = [light_sentence(ctx.lights)] if ctx.lights else []
            say = " ".join(lead + kept + ["I can't judge that for you, so please check for yourself."])
            return self._result(say, calls, retrieved, t0, {"route": "safety_describe", **gen})

        # 3. Plain questions about the road: look and answer, no tool list, no examples to copy.
        if not command_intent(q):
            say, gen = self._plain_answer(q, image, context=self._plain_context(ctx))
            if not say.strip() or _looks_like_json(say):
                say = "Sorry, I'm not sure."
            say = ground_light_claims(say, ctx.lights)
            return self._result(say, calls, retrieved, t0, {"route": "look_and_answer", **gen})

        # 4. Car commands: the tool prompt (the memory tool isn't offered).
        system = SYSTEM_PROMPT + "\n\n" + JSON_INSTRUCTIONS.format(tools=tools_prompt_block(frozenset({"recall_scene"})))
        user = f"{ctx.context_block()}\n\nDriver says: \"{q}\""
        raw, stats = self._generate(system, user, image, max_tokens=200)
        parsed = _extract_json(raw)
        if parsed is None:  # model ignored the format; fine if it's plain speech (checked below)
            parsed = {"tool_calls": [], "say": raw.strip().strip('"')}
        say = str(parsed.get("say") or "").strip()

        for call in parsed.get("tool_calls") or []:
            name, args = call.get("name"), call.get("args") or {}
            if name not in TOOL_NAMES or name == "recall_scene":
                continue
            result = execute_vehicle_tool(ctx.car, name, args)
            calls.append({"name": name, "args": args, "result": result})
            if not say:
                say = result

        # The safety envelope has the final word: if a tool clamped/blocked/failed, say what
        # actually happened instead of what the model *predicted* would happen.
        notes = [c["result"] for c in calls if any(w in c["result"] for w in ("blocked", "outside", "Error"))]
        if notes:
            say = " ".join(notes)
        if not say or _looks_like_json(say):
            # Broken or empty output: never read JSON aloud (seen in the eval: the model invented
            # a "say" tool, ran out of tokens, and the raw JSON became the answer).
            say, gen = self._plain_answer(q, image)
            stats["fallback"] = gen
        return self._result(say, calls, retrieved, t0, {"route": "command", "raw": raw[:400], **stats})
