from __future__ import annotations

import io
import re
from dataclasses import dataclass, field

from PIL import Image

from drivemind.brain.tools import CarState

SYSTEM_PROMPT = """You are DriveMind, the voice assistant built into this car. You can see the road through the front camera (the attached image is the current view) and you control vehicle functions through tools.

How to respond:
- Your reply is spoken aloud to someone who is driving. Use one or two short, natural sentences. No markdown, lists, or emoji.
- Ground visual answers in the image and the perception summary. If something isn't clearly visible, say so; never invent details like sign text you can't read.
- Visual memory is only for things that are no longer visible ("what did that sign say?"). Anything about what is visible now is answered from the current image.
- Traffic-light colours in the perception summary come from a detector that reads the lit lamp. Trust them over the small lights in the image, and if a light is "not readable", say you can't make out its colour.
- When the user asks for a vehicle function, call the tool and then confirm briefly.
- You are not a safety system. If asked whether it is safe to turn, merge, or proceed, describe what you see but tell the driver to check for themselves.
- If the driver workload is high, answer in under ten words."""


# Deciding when to search visual memory is a rule, not the model's call. The on-device 3B
# model searched memory for "Which car is on my left?" in a live demo (answering from a frame
# 17 s old) and skipped it for "Which car was on my left a moment ago?" in
# eval/brain_qa_eval.py. Memory is for questions that point to the past AND to something visible.
_PAST = re.compile(
    r"\b(did|was|were|passed|went by|drove|drive past|driving past|earlier|ago|back there|behind us|"
    r"previous|just saw|missed|said|saw|showed|remind me|last (?:sign|light|exit|car|truck|one|turn|street|building|store))\b",
    re.I,
)
_VISUAL = re.compile(
    r"\b(signs?|lights?|cars?|trucks?|bus|vans?|bikes?|bicycles?|motorcycles?|person|people|pedestrians?|"
    r"buildings?|stores?|shops?|restaurants?|exits?|streets?|roads?|intersections?|billboards?|plates?|colou?r|"
    r"written|see|saw|pass|passed|past|went by|drove by)\b",
    re.I,
)


def memory_intent(query: str) -> bool:
    """Should this question be answered from visual memory (a frame from the last few minutes)?"""
    return bool(_PAST.search(query) and _VISUAL.search(query))


# Car functions go through the tool prompt; everything else gets a plain "look and answer"
# prompt. Mixing them failed both ways on the 3B model: with example answers in the prompt it
# copied them ("Is the light green?" -> "Yes, the light ahead is green." at a RED light), and
# without them it called tools for plain questions ("Navigating to the red light").
_COMMAND = re.compile(
    r"\b(temperature|degrees|heat|heater|heating|a/?c|air ?con\w*|fan|defrost\w*|navigate|navigation|"
    r"directions|take me|drive me|route to|play|music|songs?|podcast|radio|volume|louder|quieter|mute|"
    r"windows?|seat heaters?|heated seats?|seat warmers?|wipers?|cold|freezing|chilly|warm (?:it|me)|"
    r"cool (?:it|me)|too hot|hot in here|stuffy)\b",
    re.I,
)
# Go/no-go judgments are the driver's call: the assistant describes, never decides.
_SAFETY = re.compile(
    r"\b(?:safe|ok|okay|clear) to\b|\bis it (?:safe|clear)\b|\bgood to go\b|"
    r"\b(?:can|should|could) i (?:go|turn|merge|pass|cross|change lanes|pull out|proceed)\b",
    re.I,
)


def command_intent(query: str) -> bool:
    return bool(_COMMAND.search(query))


def safety_intent(query: str) -> bool:
    return bool(_SAFETY.search(query))


# Traffic-light colour comes from the lamp detector (perception/road.py), not from a language model
# squinting at a few pixels: at 448 px the on-device model called two green lights red.
_LIGHT = re.compile(r"\b(?:traffic )?(?:lights?|signals?)\b", re.I)
_NOT_TRAFFIC = re.compile(r"\b(?:brake|tail|head|street|hazard|fog|turn|dashboard|warning|interior|reading)\s*(?:lights?|signals?)\b", re.I)
_LIGHT_STATE = re.compile(r"\b(?:red|green|yellow|amber|colou?r|changed?|turned)\b", re.I)
_COLOURS = re.compile(r"\b(red|green|yellow|amber)\b", re.I)


def light_intent(query: str) -> bool:
    """A question about what colour the traffic light is showing."""
    return bool(_LIGHT.search(query) and _LIGHT_STATE.search(query) and not _NOT_TRAFFIC.search(query))


def light_sentence(lights: list[str | None]) -> str:
    seen = [c for c in ("red", "yellow", "green") if c in lights]
    if not lights:
        return "I don't see a traffic light right now."
    if not seen:
        return "I can see a traffic light, but I can't make out its colour from here."
    if len(seen) > 1:
        return f"I can see {' and '.join(seen)} lights, so check which one is yours."
    return f"The light ahead is {seen[0]}."


def answer_light_question(query: str, lights: list[str | None] | None) -> str | None:
    """Deterministic answer to a traffic-light colour question, or None if this isn't one (or there's
    no road frame). Go/no-go and past-tense questions take their own routes."""
    if lights is None or not light_intent(query) or safety_intent(query) or memory_intent(query):
        return None
    say = light_sentence(lights)
    seen = [c for c in ("red", "yellow", "green") if c in lights]
    asked = {"yellow" if c.lower() == "amber" else c.lower() for c in _COLOURS.findall(query)}
    if len(seen) == 1 and len(asked) == 1 and re.match(r"\s*(?:is|are)\b", query, re.I):
        return ("Yes, the light ahead is " if asked == set(seen) else "No, the light ahead is ") + f"{seen[0]}."
    return say


# A colour within a few words of "light"/"signal" ("the light ahead is red", "red lights ahead"), not
# "the car with red tail lights".
_LIGHT_CLAIM = re.compile(
    r"\b(?:light|signal)s?\b(?:\W+\w+){0,3}?\W+(?:red|green|yellow|amber)\b"
    r"|\b(?:red|green|yellow|amber)\b(?:\W+\w+){0,2}?\W+(?:light|signal)s?\b",
    re.I,
)


def mentions_light_colour(sentence: str) -> bool:
    return bool(_LIGHT_CLAIM.search(_NOT_TRAFFIC.sub(" ", sentence)))


def ground_light_claims(say: str, lights: list[str | None] | None) -> str:
    """Replace whatever a language model said about traffic-light colour with the detector's reading
    (guardrails are code). Sentences that mention a light and a colour go; the detector's sentence
    comes first."""
    if lights is None:
        return say
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", say.strip()) if s]
    kept = [s for s in sentences if not mentions_light_colour(s)]
    if len(kept) == len(sentences):
        return say
    return " ".join([light_sentence(lights)] + kept)


@dataclass
class BrainContext:
    query: str
    road_jpeg: bytes | None
    road_summary: str
    driver_summary: str
    car: CarState
    workload: str
    history: list[tuple[str, str]] = field(default_factory=list)
    lights: list[str | None] | None = None  # lit colour per traffic light in view; None = no road frame

    def context_block(self) -> str:
        hist = "\n".join(f"Driver: {u}\nYou: {a}" for u, a in self.history[-4:]) or "(none)"
        c = self.car
        return (
            f"[Live context]\nRoad perception: {self.road_summary}\n{self.driver_summary}\n"
            f"Driver workload: {self.workload}. Speed: {c.speed_mph:.0f} mph.\n"
            f"Car: {c.temperature_f:g}°F, fan {c.fan_level}, defrost {'on' if c.defrost else 'off'}, "
            f"media: {c.media or 'off'}, navigation: {c.destination or 'none'}.\n"
            f"[Recent conversation]\n{hist}"
        )


@dataclass
class BrainResult:
    text: str
    brain: str
    tool_calls: list[dict] = field(default_factory=list)
    retrieved: list[dict] = field(default_factory=list)
    ms: float = 0.0
    detail: dict = field(default_factory=dict)


def downscale_jpeg(jpeg: bytes, width: int) -> Image.Image:
    """Fewer pixels -> fewer vision tokens -> faster prefill. For a ViT encoder with 14 px
    patches (and 2x2 merging in Qwen2.5-VL), tokens scale with area: 448 px wide is ~4x
    fewer tokens than 896."""
    img = Image.open(io.BytesIO(jpeg)).convert("RGB")
    if img.width > width:
        img = img.resize((width, round(img.height * width / img.width)), Image.BILINEAR)
    return img


def to_jpeg(img: Image.Image, quality: int = 80) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()
