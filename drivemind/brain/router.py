"""Hybrid edge/cloud routing.

Concepts:

* **Model cascades / routing.** Most in-car requests ("I'm cold") are easy, and a 3B
  on-device model handles them in well under a second with no network. Hard ones ("what
  did that sign say a minute ago?", "explain why the car beeped") benefit from a frontier
  model. A router sends each query to the cheapest model likely to get it right.
* **Graceful degradation.** Cars drive through tunnels. If the cloud is unreachable, we
  fall back to the local brain instead of failing. This is the availability argument for edge AI.
* Here the router is a transparent heuristic so you can reason about it. A natural next
  step (see docs/CONCEPTS.md) is a *learned* router: a small classifier trained on which
  queries the local model gets wrong.
"""

from __future__ import annotations

import re
import time

from drivemind.brain.claude_brain import ClaudeUnavailable
from drivemind.brain.common import BrainContext, BrainResult, answer_light_question

COMPLEX = re.compile(
    r"\b(why|explain|compare|describe|what (did|was)|earlier|ago|passed|just saw|read|how far|"
    r"recommend|should i|summar|story|tell me about)\b",
    re.I,
)


class BrainRouter:
    def __init__(self, local, claude) -> None:
        self.local = local
        self.claude = claude
        self.claude_down_until = 0.0

    def available(self) -> dict:
        return {"local": self.local is not None, "claude": self.claude is not None}

    def route(self, ctx: BrainContext, mode: str) -> str:
        if answer_light_question(ctx.query, ctx.lights) is not None:
            return "perception"  # traffic-light colour: the lamp detector answers, in any mode
        claude_ok = self.claude is not None and time.monotonic() > self.claude_down_until
        if mode == "claude" and claude_ok:
            return "claude"
        if mode == "local" and self.local is not None:
            return "local"
        if self.local is None:
            return "claude"
        # auto: complex or long queries go to the cloud when it's reachable
        if claude_ok and (COMPLEX.search(ctx.query) or len(ctx.query.split()) > 14):
            return "claude"
        return "local"

    def answer(self, ctx: BrainContext, mode: str, memory=None) -> BrainResult:
        choice = self.route(ctx, mode)
        if choice == "perception":
            t0 = time.perf_counter()
            text = answer_light_question(ctx.query, ctx.lights)
            return BrainResult(text=text, brain="perception", ms=round((time.perf_counter() - t0) * 1000, 1),
                               detail={"route": "traffic_light", "lights": ctx.lights})
        if choice == "claude":
            try:
                res = self.claude.answer(ctx, memory)
                res.detail["routed"] = "auto" if mode == "auto" else "forced"
                return res
            except ClaudeUnavailable as e:
                self.claude_down_until = time.monotonic() + 30  # circuit breaker
                if self.local is None:
                    return BrainResult(text="I can't reach the cloud right now.", brain="none", detail={"error": str(e)})
                res = self.local.answer(ctx, memory)
                res.detail["fallback_reason"] = str(e)[:200]
                return res
        return self.local.answer(ctx, memory)
