"""One connected dashboard = one Session. This is the real-time orchestration layer.

Concepts:

* **Pipeline parallelism.** Road perception, driver monitoring, audio, memory and the
  brain each run on their own worker thread, so a slow VLM answer never stalls collision
  warnings. Independent stages, independent latency budgets.
* **Backpressure: latest-frame-wins.** If the road worker is still busy when a new frame
  arrives, we *drop* the new one rather than queueing it. Queues in a real-time system
  turn into latency: a 1-second-old collision warning is useless. We report the drop
  rate so you can see when a stage is over budget.
* **Binary WebSocket protocol.** Frames and audio go as raw bytes with a 1-byte type
  header (no base64 inflation); control messages and results are small JSON.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from fastapi import WebSocket

from drivemind.brain.common import BrainContext
from drivemind.brain.safety import AttentionManager
from drivemind.brain.tools import CarState
from drivemind.perception.driver import DriverMonitor, summarize_driver
from drivemind.perception.road import RoadPerception, light_states, summarize_road
from drivemind.server.models import models

log = logging.getLogger("drivemind.session")

MSG_ROAD, MSG_CABIN, MSG_AUDIO = 1, 2, 3

# MLX (Whisper + VLM) must be serialized; one shared single-thread executor does that.
MLX_EXECUTOR = ThreadPoolExecutor(1, thread_name_prefix="mlx")
# A perception stage busy for longer than this is stalled (normal: 5-20 ms per frame).
STALL_S = 5.0
# Monitors abandoned by the watchdog are kept referenced forever: if the garbage collector
# finalized one, MediaPipe's __del__ could deadlock some *other* thread (see DriverMonitor.close).
_ABANDONED: list = []


class RateMeter:
    """Exponential moving average of a stage's throughput (FPS)."""

    def __init__(self) -> None:
        self.last: float | None = None
        self.fps = 0.0

    def tick(self) -> float:
        now = time.perf_counter()
        if self.last is not None:
            inst = 1.0 / max(now - self.last, 1e-3)
            self.fps = inst if self.fps == 0 else 0.85 * self.fps + 0.15 * inst
        self.last = now
        return round(self.fps, 1)


def _decode(jpeg: bytes) -> np.ndarray | None:
    return cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)


class Session:
    def __init__(self, ws: WebSocket) -> None:
        self.ws = ws
        self.loop = asyncio.get_running_loop()
        self.ex = {k: ThreadPoolExecutor(1, thread_name_prefix=k) for k in ("road", "cabin", "audio", "memory", "brain")}
        self.road = None  # created lazily on the worker thread
        self.driver = None
        self.vad = None
        self.car = CarState()
        self.attn = AttentionManager()
        self.brain_mode = "auto"
        self.listen_mode = "ptt"  # ptt | handsfree
        self.ptt_down = False
        self._ptt_buf: list[bytes] = []
        self.tts_speaking = False
        self.busy = {"road": False, "cabin": False, "memory": False, "brain": False}
        self.busy_since = {"road": 0.0, "cabin": 0.0}
        self.dropped = {"road": 0, "cabin": 0}
        self.meters = {"road": RateMeter(), "cabin": RateMeter()}
        self.last_road: dict | None = None
        self.last_road_jpeg: bytes | None = None
        self.last_driver: dict | None = None
        self.history: deque[tuple[str, str]] = deque(maxlen=6)
        self.send_lock = asyncio.Lock()
        self.closed = False

    # ------------------------------------------------------------------ io
    async def send(self, msg: dict) -> None:
        if self.closed:
            return
        try:
            text = json.dumps(msg)
        except (TypeError, ValueError):
            log.exception("unserializable %s message", msg.get("type"))
            return
        async with self.send_lock:
            try:
                await self.ws.send_text(text)
            except Exception:  # noqa: BLE001 - socket gone
                self.closed = True

    def send_threadsafe(self, msg: dict) -> None:
        asyncio.run_coroutine_threadsafe(self.send(msg), self.loop)

    async def start(self) -> None:
        def on_status(status):
            self.send_threadsafe({"type": "status", "models": status})

        models.listeners.append(on_status)
        self._status_cb = on_status
        await self.send({"type": "status", "models": dict(models.status)})
        await self.send({"type": "car", "state": self.car.to_dict()})
        if models.memory:
            models.memory.clear()
        # Build this session's perception models now (each on its own worker thread), not on
        # the first frame: the first frames queue behind this instead of paying the load time.
        self.ex["road"].submit(self._init_road)
        self.ex["cabin"].submit(self._init_driver)

    def _init_road(self) -> None:
        if self.road is None and not self.closed:
            self.road = RoadPerception()

    def _init_driver(self) -> None:
        if self.driver is None and not self.closed:
            self.driver = DriverMonitor()

    def close(self) -> None:
        self.closed = True
        if getattr(self, "_status_cb", None) in models.listeners:
            models.listeners.remove(self._status_cb)
        for ex in self.ex.values():
            ex.shutdown(wait=False, cancel_futures=True)
        # Release the face model deterministically, on a helper thread that first waits for
        # the cabin worker's in-flight frame to finish (see DriverMonitor.close for why).
        driver, self.driver = self.driver, None
        if driver is not None:
            threading.Thread(target=self._dispose, args=(self.ex["cabin"], driver), daemon=True).start()

    @staticmethod
    def _dispose(executor: ThreadPoolExecutor, driver: DriverMonitor) -> None:
        executor.shutdown(wait=True)
        driver.close()

    # ------------------------------------------------------------- routing
    async def on_bytes(self, data: bytes) -> None:
        if not data:
            return
        kind, payload = data[0], data[1:]
        if kind == MSG_ROAD:
            await self._spawn("road", self._road_frame, payload)
        elif kind == MSG_CABIN:
            await self._spawn("cabin", self._cabin_frame, payload)
        elif kind == MSG_AUDIO:
            await self._audio(payload)

    async def _spawn(self, stage: str, fn, payload) -> None:
        if self.busy[stage]:
            if time.monotonic() - self.busy_since[stage] < STALL_S:
                self.dropped[stage] += 1  # latest-frame-wins backpressure
                return
            # Watchdog: the worker is stuck. Abandon it (a Python thread can't be killed) and
            # start a fresh worker + model so the safety-relevant stage keeps running.
            log.error("%s stage stalled for >%.0fs; replacing its worker", stage, STALL_S)
            await self.send({"type": "error", "stage": stage, "message": f"{stage} stage stalled; restarted"})
            self.ex[stage] = ThreadPoolExecutor(1, thread_name_prefix=stage)
            if stage == "road":
                self.road = None
            else:
                _ABANDONED.append(self.driver)
                self.driver = None
        self.busy[stage] = True
        self.busy_since[stage] = time.monotonic()
        asyncio.create_task(fn(payload))

    async def on_json(self, msg: dict) -> None:
        t = msg.get("type")
        if t == "brain":
            self.brain_mode = msg.get("mode", "auto")
        elif t == "listen_mode":
            self.listen_mode = msg.get("mode", "ptt")
            if self.vad:
                self.vad.reset()
        elif t == "ptt":
            await self._ptt(bool(msg.get("down")))
        elif t == "tts":
            self.tts_speaking = bool(msg.get("speaking"))
        elif t == "query":
            text = str(msg.get("text", "")).strip()
            if text:
                asyncio.create_task(self._handle_query(text, {"source": "typed"}))
        elif t == "speed":
            self.car.speed_mph = float(msg.get("mph", 0))
        elif t == "reset":
            which = msg.get("which")
            if which == "road" and self.road:
                await self.loop.run_in_executor(self.ex["road"], self.road.reset)
                if models.memory:
                    models.memory.clear()
            if which == "cabin" and self.driver:
                await self.loop.run_in_executor(self.ex["cabin"], self.driver.reset)

    # ---------------------------------------------------------- perception
    async def _road_frame(self, jpeg: bytes) -> None:
        try:
            def work():
                if self.road is None:
                    self.road = RoadPerception()
                frame = _decode(jpeg)
                return None if frame is None else self.road.process(frame)

            res = await self.loop.run_in_executor(self.ex["road"], work)
            if res is None:
                return
            self.last_road, self.last_road_jpeg = res, jpeg
            res.update(type="road", fps=self.meters["road"].tick(), dropped=self.dropped["road"])
            await self.send(res)
            await self._safety_tick()
            if models.memory and not self.busy["memory"]:
                self.busy["memory"] = True
                asyncio.create_task(self._remember(jpeg, summarize_road(res)))
        except Exception as e:  # noqa: BLE001
            log.exception("road stage failed")
            await self.send({"type": "error", "stage": "road", "message": str(e)[:200]})
        finally:
            self.busy["road"] = False

    async def _remember(self, jpeg: bytes, summary: str) -> None:
        try:
            info = await self.loop.run_in_executor(self.ex["memory"], models.memory.observe, jpeg, summary)
            if info.get("stored"):
                await self.send({"type": "memory", **info})
        except Exception:  # noqa: BLE001
            log.exception("memory stage failed")
        finally:
            self.busy["memory"] = False

    async def _cabin_frame(self, jpeg: bytes) -> None:
        try:
            def work():
                if self.closed:
                    return None  # don't build a new monitor for a session that just ended
                if self.driver is None:
                    self.driver = DriverMonitor()
                frame = _decode(jpeg)
                return None if frame is None else self.driver.process(frame)

            res = await self.loop.run_in_executor(self.ex["cabin"], work)
            if res is None:
                return
            self.last_driver = res
            res.update(type="cabin", fps=self.meters["cabin"].tick(), dropped=self.dropped["cabin"])
            await self.send(res)
            await self._safety_tick()
        except Exception as e:  # noqa: BLE001
            log.exception("cabin stage failed")
            await self.send({"type": "error", "stage": "cabin", "message": str(e)[:200]})
        finally:
            self.busy["cabin"] = False

    async def _safety_tick(self) -> None:
        workload = self.attn.workload(self.last_road, self.last_driver)
        for alert in self.attn.proactive_alerts(self.last_road, self.last_driver):
            await self.send({"type": "alert", **alert})
        for reply in self.attn.release_deferred(workload):
            await self.send(reply)

    # --------------------------------------------------------------- audio
    async def _ensure_vad(self) -> None:
        if self.vad is None:
            from drivemind.audio.vad import StreamingVAD
            self.vad = await self.loop.run_in_executor(self.ex["audio"], StreamingVAD)

    async def _audio(self, pcm: bytes) -> None:
        listening = (self.listen_mode == "handsfree" and not self.tts_speaking) or self.ptt_down
        if not listening:
            return
        await self._ensure_vad()
        if self.listen_mode == "ptt":
            # Push-to-talk: buffer everything while held; the VAD still reports speech activity.
            events = await self.loop.run_in_executor(self.ex["audio"], self.vad.feed, pcm)
            self._ptt_buf.append(pcm)
            if any(e["type"] == "speech_start" for e in events):
                await self.send({"type": "vad", "speaking": True})
            return
        events = await self.loop.run_in_executor(self.ex["audio"], self.vad.feed, pcm)
        for e in events:
            if e["type"] == "speech_start":
                await self.send({"type": "vad", "speaking": True})
            elif e["type"] == "speech_cancel":
                await self.send({"type": "vad", "speaking": False})
            elif e["type"] == "utterance":
                await self.send({"type": "vad", "speaking": False})
                asyncio.create_task(self._handle_utterance(e["audio"]))

    async def _ptt(self, down: bool) -> None:
        await self._ensure_vad()
        if down:
            self.ptt_down = True
            self._ptt_buf = []
            await self.loop.run_in_executor(self.ex["audio"], self.vad.reset)
            return
        self.ptt_down = False
        await self.send({"type": "vad", "speaking": False})
        pcm = b"".join(self._ptt_buf)
        self._ptt_buf = []
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        if audio.size < 16_000 * 0.3:
            return
        asyncio.create_task(self._handle_utterance(audio))

    async def _handle_utterance(self, audio: np.ndarray) -> None:
        if models.asr is None:
            await self.send({"type": "reply", "text": "Speech recognition is still loading.", "brain": "system"})
            return
        t_end = time.perf_counter()
        asr = await self.loop.run_in_executor(MLX_EXECUTOR, models.asr.transcribe, audio)
        await self.send({"type": "transcript", "text": asr["text"], "ms": asr["ms"], "audio_s": round(audio.size / 16_000, 2)})
        if asr["text"]:
            await self._handle_query(asr["text"], {"asr_ms": asr["ms"], "t_end": t_end})

    # --------------------------------------------------------------- brain
    async def _handle_query(self, text: str, timing: dict) -> None:
        if models.router is None:
            await self.send({"type": "reply", "text": "My brain is still loading. Give me a moment.", "brain": "system"})
            return
        if self.busy["brain"]:
            await self.send({"type": "reply", "text": "One sec, still thinking about your last question.", "brain": "system"})
            return
        self.busy["brain"] = True
        try:
            workload = self.attn.workload(self.last_road, self.last_driver)
            ctx = BrainContext(
                query=text,
                road_jpeg=self.last_road_jpeg,
                road_summary=summarize_road(self.last_road),
                driver_summary=summarize_driver(self.last_driver),
                car=self.car,
                workload=workload,
                history=list(self.history),
                lights=light_states(self.last_road),
            )
            await self.send({"type": "thinking", "brain": models.router.route(ctx, self.brain_mode)})
            ex = MLX_EXECUTOR if models.router.route(ctx, self.brain_mode) == "local" else self.ex["brain"]
            res = await self.loop.run_in_executor(ex, models.router.answer, ctx, self.brain_mode, models.memory)
            self.history.append((text, res.text))
            total = round((time.perf_counter() - timing["t_end"]) * 1000, 1) if "t_end" in timing else None
            reply = {
                "type": "reply",
                "query": text,
                "text": res.text,
                "brain": res.brain,
                "tool_calls": res.tool_calls,
                "retrieved": res.retrieved,
                "latency": {"asr_ms": timing.get("asr_ms"), "brain_ms": res.ms, "total_ms": total},
                "detail": res.detail,
                "workload": workload,
            }
            gated = self.attn.gate_reply(reply, self.attn.workload(self.last_road, self.last_driver))
            if gated is None:
                await self.send({"type": "deferred", "query": text})
            else:
                await self.send(gated)
            if res.tool_calls:
                await self.send({"type": "car", "state": self.car.to_dict()})
        except Exception as e:  # noqa: BLE001
            log.exception("brain failed")
            await self.send({"type": "reply", "text": "Sorry, something went wrong.", "brain": "error", "detail": {"error": str(e)[:300]}})
        finally:
            self.busy["brain"] = False
