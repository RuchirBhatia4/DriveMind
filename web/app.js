// DriveMind dashboard: captures camera frames + mic audio, streams them to the server
// over one WebSocket, and renders perception results, driver state and the assistant.

const MSG_ROAD = 1, MSG_CABIN = 2, MSG_AUDIO = 3;
const ROAD_FPS = 15, CABIN_FPS = 15;
const $ = (id) => document.getElementById(id);

// ------------------------------------------------------------------ socket
let ws;
function connect() {
  ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => {
    $("conn").textContent = "live"; $("conn").classList.add("on");
    sendJSON({ type: "brain", mode: $("brain-mode").value });
    sendJSON({ type: "listen_mode", mode: $("listen-mode").value });
    sendJSON({ type: "speed", mph: +$("speed").value });
  };
  ws.onclose = () => { $("conn").textContent = "reconnecting…"; $("conn").classList.remove("on"); setTimeout(connect, 1000); };
  ws.onmessage = (e) => handle(JSON.parse(e.data));
}
const sendJSON = (o) => ws && ws.readyState === 1 && ws.send(JSON.stringify(o));
function sendBinary(kind, buf) {
  // Client-side backpressure: if the socket is backed up, skip this frame entirely.
  if (!ws || ws.readyState !== 1 || ws.bufferedAmount > 1_500_000) return false;
  const out = new Uint8Array(buf.byteLength + 1);
  out[0] = kind; out.set(new Uint8Array(buf), 1);
  ws.send(out);
  return true;
}

// ------------------------------------------------------------------ cameras
const sources = {
  road: { video: $("road-video"), overlay: $("road-overlay"), kind: MSG_ROAD, fps: ROAD_FPS, width: 640, stream: null, timer: null },
  cabin: { video: $("cabin-video"), overlay: $("cabin-overlay"), kind: MSG_CABIN, fps: CABIN_FPS, width: 480, stream: null, timer: null },
};

async function listCameras() {
  try { (await navigator.mediaDevices.getUserMedia({ video: true })).getTracks().forEach((t) => t.stop()); } catch {}
  const cams = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput");
  for (const which of ["road", "cabin"]) {
    const sel = $(`${which}-source`);
    sel.innerHTML = "";
    sel.append(new Option(which === "road" ? "— none —" : "— off —", ""));
    cams.forEach((c, i) => sel.append(new Option(c.label || `Camera ${i + 1}`, c.deviceId)));
    sel.onchange = () => startCamera(which, sel.value);
  }
  // Default: the built-in webcam watches the driver; the road uses a dashcam video file.
  if (cams.length) { $("cabin-source").value = cams[0].deviceId; startCamera("cabin", cams[0].deviceId); }
  if (cams.length > 1) { $("road-source").value = cams[1].deviceId; startCamera("road", cams[1].deviceId); }
}

function stopSource(which) {
  const s = sources[which];
  clearInterval(s.timer);
  if (s.stream) s.stream.getTracks().forEach((t) => t.stop());
  s.stream = null;
  s.video.srcObject = null;
  s.video.removeAttribute("src");
}

async function startCamera(which, deviceId) {
  stopSource(which);
  sendJSON({ type: "reset", which });
  if (!deviceId) { if (which === "road") $("road-placeholder").style.display = ""; return; }
  const s = sources[which];
  s.stream = await navigator.mediaDevices.getUserMedia({ video: { deviceId: { exact: deviceId }, width: { ideal: 1280 }, height: { ideal: 720 } } });
  s.video.srcObject = s.stream;
  s.video.play().catch(() => {});
  beginCapture(which);
}

$("road-file").onchange = async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  stopSource("road");
  sendJSON({ type: "reset", which: "road" });
  $("road-source").value = "";
  const v = sources.road.video;
  v.src = URL.createObjectURL(f);
  // play() can reject (autoplay policy, or a hidden tab pausing video to save power);
  // capture must not depend on it. We retry on visibility change below.
  v.play().catch(() => {});
  beginCapture("road");
};

function beginCapture(which) {
  const s = sources[which];
  if (which === "road") $("road-placeholder").style.display = "none";
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  let inflight = false;
  s.timer = setInterval(() => {
    const v = s.video;
    if (inflight || v.readyState < 2 || !v.videoWidth) return;
    canvas.width = s.width;
    canvas.height = Math.round((s.width * v.videoHeight) / v.videoWidth);
    ctx.drawImage(v, 0, 0, canvas.width, canvas.height);
    inflight = true;
    canvas.toBlob(async (blob) => {
      if (blob) sendBinary(s.kind, await blob.arrayBuffer());
      inflight = false;
    }, "image/jpeg", 0.72);
  }, 1000 / s.fps);
}

// Map normalized [0..1] coords onto the letterboxed video area inside the overlay canvas.
function videoRect(s) {
  const c = s.overlay, v = s.video;
  const W = (c.width = c.clientWidth * devicePixelRatio), H = (c.height = c.clientHeight * devicePixelRatio);
  if (!v.videoWidth) return { x: 0, y: 0, w: W, h: H };
  const scale = Math.min(W / v.videoWidth, H / v.videoHeight);
  const w = v.videoWidth * scale, h = v.videoHeight * scale;
  return { x: (W - w) / 2, y: (H - h) / 2, w, h };
}

// ------------------------------------------------------------------ rendering
const COLORS = { car: "--c-car", truck: "--c-car", bus: "--c-car", motorcycle: "--c-car", person: "--c-person", bicycle: "--c-person", "traffic light": "--c-sign", "stop sign": "--c-sign" };
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

function drawRoad(r) {
  const s = sources.road, ctx = s.overlay.getContext("2d"), R = videoRect(s), dpr = devicePixelRatio;
  ctx.clearRect(0, 0, s.overlay.width, s.overlay.height);
  // Assumed heading (image centre). An object is "in path" when its centre is within ~0.7 of
  // its own width of this line, a test that doesn't change with distance (see road.py).
  ctx.setLineDash([6 * dpr, 6 * dpr]); ctx.strokeStyle = "rgba(255,255,255,.3)"; ctx.lineWidth = 1.5 * dpr;
  ctx.beginPath(); ctx.moveTo(R.x + 0.5 * R.w, R.y + 0.45 * R.h); ctx.lineTo(R.x + 0.5 * R.w, R.y + R.h); ctx.stroke();
  ctx.setLineDash([]);
  ctx.font = `${12 * dpr}px ui-monospace, Menlo, monospace`;
  for (const t of r.tracks) {
    const [x1, y1, x2, y2] = t.box;
    const x = R.x + x1 * R.w, y = R.y + y1 * R.h, w = (x2 - x1) * R.w, h = (y2 - y1) * R.h;
    const threat = t.id === r.lead_id && r.fcw !== "none";
    const color = threat ? css("--crit") : css(COLORS[t.label] || "--c-car");
    ctx.strokeStyle = color; ctx.lineWidth = (threat ? 4 : 2) * dpr;
    ctx.strokeRect(x, y, w, h);
    let label = `${t.label}${t.id != null ? " #" + t.id : ""}`;
    if (t.state) label += ` ${t.state.toUpperCase()}`;
    if (t.ttc != null && t.in_path) label += ` TTC ${t.ttc}s`;
    const tw = ctx.measureText(label).width + 8 * dpr;
    ctx.fillStyle = color; ctx.fillRect(x, y - 16 * dpr, tw, 16 * dpr);
    ctx.fillStyle = "#000"; ctx.fillText(label, x + 4 * dpr, y - 4 * dpr);
  }
  $("road-fps").textContent = `${r.fps} fps`;
  $("road-ms").textContent = `${r.ms} ms`;
  $("road-drop").textContent = `${r.dropped} dropped`;
  const fcw = $("fcw");
  fcw.className = `fcw ${r.fcw}`;
  fcw.textContent = r.fcw === "critical" ? `⚠ BRAKE · TTC ${r.min_ttc}s` : r.fcw === "warning" ? `Closing vehicle · TTC ${r.min_ttc}s` : "";
}

const earHist = [];
function drawCabin(c) {
  const s = sources.cabin, ctx = s.overlay.getContext("2d"), R = videoRect(s), dpr = devicePixelRatio;
  ctx.clearRect(0, 0, s.overlay.width, s.overlay.height);
  const stateColor = { alert: css("--ok"), drowsy: css("--warn"), distracted: css("--warn"), microsleep: css("--crit") }[c.state] || "#9ca3af";
  if (c.eyes) {
    ctx.fillStyle = stateColor;
    for (const [px, py] of c.eyes) { ctx.beginPath(); ctx.arc(R.x + px * R.w, R.y + py * R.h, 2.2 * dpr, 0, 7); ctx.fill(); }
  }
  $("cabin-fps").textContent = `${c.fps} fps`;
  $("cabin-ms").textContent = `${c.ms} ms`;
  const d = $("dstate");
  d.className = `dstate ${c.state}`;
  d.textContent = c.state === "calibrating" ? `calibrating ${Math.round((c.calib_progress || 0) * 100)}% · look ahead` : c.state.replace("_", " ");
  if (c.perclos != null) {
    $("perclos").textContent = `${(c.perclos * 100).toFixed(0)}%`;
    const bar = $("perclos-bar");
    bar.style.width = `${Math.min(100, (c.perclos / 0.3) * 100)}%`;
    bar.style.background = c.perclos > 0.15 ? css("--crit") : c.perclos > 0.08 ? css("--warn") : css("--ok");
  }
  $("yawns").textContent = c.yawns_2min ?? "–";
  $("yaw").textContent = c.yaw != null ? `${c.yaw}°` : "–";
  $("pitch").textContent = c.pitch != null ? `${c.pitch}°` : "–";
  $("closed").textContent = c.closed_for != null ? `${c.closed_for}s` : "–";
  // Plot what actually drives the alerts: the eye-closure probability (trained model or
  // MediaPipe blendshape). Before the model has a decision, fall back to raw EAR.
  if (c.eye_prob != null) {
    $("spark-label").textContent = c.eye_source === "trained_gbm" ? "Eye closure probability (trained model)" : "Eye closure (MediaPipe blink score)";
    earHist.push([c.eye_prob, c.eye_threshold, 1.0]);
  } else if (c.ear != null) {
    $("spark-label").textContent = "Eye aspect ratio";
    earHist.push([c.ear, c.ear_threshold, 0.45]);
  }
  if (earHist.length > 150) earHist.shift();
  drawSpark();
}

function drawSpark() {
  const cv = $("ear-spark"), ctx = cv.getContext("2d");
  cv.width = cv.clientWidth * devicePixelRatio; cv.height = cv.clientHeight * devicePixelRatio;
  const W = cv.width, H = cv.height, max = earHist.at(-1)?.[2] || 0.45, y = (v) => H - (Math.min(v, max) / max) * H;
  ctx.clearRect(0, 0, W, H);
  const th = earHist.at(-1)?.[1];
  if (th) { ctx.strokeStyle = css("--crit"); ctx.setLineDash([4, 4]); ctx.beginPath(); ctx.moveTo(0, y(th)); ctx.lineTo(W, y(th)); ctx.stroke(); ctx.setLineDash([]); }
  ctx.strokeStyle = css("--accent"); ctx.lineWidth = 2 * devicePixelRatio; ctx.beginPath();
  earHist.forEach(([v], i) => { const x = (i / 149) * W; i ? ctx.lineTo(x, y(v)) : ctx.moveTo(x, y(v)); });
  ctx.stroke();
}

// ------------------------------------------------------------------ chat + speech
function addMsg(cls, html) {
  const el = document.createElement("div");
  el.className = `msg ${cls}`; el.innerHTML = html;
  $("chat").append(el); $("chat").scrollTop = 1e9;
  return el;
}
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

function speak(text, urgent = false) {
  if (!("speechSynthesis" in window) || !text) return;
  if (urgent) speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  u.rate = urgent ? 1.15 : 1.05;
  u.onstart = () => sendJSON({ type: "tts", speaking: true });
  u.onend = u.onerror = () => sendJSON({ type: "tts", speaking: false });
  speechSynthesis.speak(u);
}

let audioCtx;
function beep(freq = 880, ms = 160, times = 2) {
  audioCtx ||= new AudioContext();
  for (let i = 0; i < times; i++) {
    const o = audioCtx.createOscillator(), g = audioCtx.createGain(), t = audioCtx.currentTime + i * (ms / 1000) * 1.6;
    o.frequency.value = freq; o.type = "square"; g.gain.value = 0.08;
    o.connect(g).connect(audioCtx.destination); o.start(t); o.stop(t + ms / 1000);
  }
}

function flashAlert(text) {
  $("alert-frame").classList.add("on");
  const b = $("alert-banner"); b.textContent = text; b.classList.add("on");
  clearTimeout(flashAlert.t);
  flashAlert.t = setTimeout(() => { $("alert-frame").classList.remove("on"); b.classList.remove("on"); }, 2200);
}

let pendingUser = null;
function handle(m) {
  switch (m.type) {
    case "road": drawRoad(m); break;
    case "cabin": drawCabin(m); break;
    case "memory": $("mem-count").textContent = `${m.size} keyframes`; break;
    case "status": renderStatus(m.models); break;
    case "car": renderCar(m.state); break;
    case "vad": $("mic").classList.toggle("vad", m.speaking); break;
    case "transcript":
      pendingUser = addMsg("user", m.text ? esc(m.text) : "<i>(didn’t catch that)</i>");
      pendingUser.insertAdjacentHTML("beforeend", `<span class="meta">${m.audio_s}s audio · Whisper ${m.ms} ms</span>`);
      break;
    case "thinking": $("brain-badge").textContent = `thinking on ${{ local: "on-device VLM", claude: "Claude", perception: "light detector" }[m.brain] || m.brain}…`; break;
    case "reply": renderReply(m); break;
    case "deferred": addMsg("sys", `Holding my answer to “${esc(m.query)}”. The road needs your attention.`); break;
    case "alert":
      if (m.level === "critical") { beep(1200, 140, 3); flashAlert(m.text); addMsg("alert", `⚠ ${esc(m.text)}`); speak(m.text, true); }
      else if (m.level === "warning") { beep(700, 160, 2); addMsg("alert", esc(m.text)); speak(m.text, true); }
      else { addMsg("engage", esc(m.text)); speak(m.text); }
      break;
    case "error": console.warn(m); break;
  }
}

function renderReply(m) {
  $("brain-badge").textContent = "";
  let meta = [];
  const brainName = { local: "on-device VLM", claude: "Claude", perception: "light detector", system: "system", error: "error" }[m.brain] || m.brain;
  meta.push(brainName);
  if (m.latency?.brain_ms) meta.push(`brain ${Math.round(m.latency.brain_ms)} ms`);
  if (m.latency?.total_ms) meta.push(`end-to-end ${Math.round(m.latency.total_ms)} ms`);
  if (m.detail?.decode_tps) meta.push(`${m.detail.decode_tps} tok/s`);
  if (m.detail?.fallback_reason) meta.push("cloud unreachable → fell back to local");
  if (m.deferred) meta.push("delivered after hazard cleared");
  if (m.shortened) meta.push("shortened: high workload");
  const chips = (m.tool_calls || []).map((c) => `<span class="chip">${esc(c.name)}(${esc(Object.values(c.args || {}).join(", "))})</span>`).join("");
  const mem = (m.retrieved || []).map((r) => `<span class="chip">memory: ${r.seconds_ago}s ago · sim ${r.similarity}</span>`).join("");
  addMsg("bot", `${esc(m.text)}${chips || mem ? "<br>" + chips + mem : ""}<span class="meta">${meta.join(" · ")}</span>`);
  speak(m.text);
  renderLatency(m.latency || {});
  for (const c of m.tool_calls || []) logTool(`${c.name}(${JSON.stringify(c.args)}) → ${c.result}`);
}

function renderLatency(l) {
  const rows = [["ASR", l.asr_ms], ["Brain", l.brain_ms], ["Total", l.total_ms]].filter(([, v]) => v != null);
  const max = Math.max(...rows.map(([, v]) => v), 1);
  $("latency").innerHTML = rows.map(([k, v]) => `<div class="lat-row"><span>${k}</span><div class="lat-bar"><div style="width:${(v / max) * 100}%"></div></div><span>${Math.round(v)} ms</span></div>`).join("");
}

function logTool(line) {
  const el = document.createElement("div");
  el.textContent = `${new Date().toLocaleTimeString()}  ${line}`;
  $("tool-log").prepend(el);
}

const STATUS_NAMES = { road: "YOLO road", asr: "Whisper", memory: "CLIP memory", local_brain: "Qwen2.5-VL 4-bit", claude: "Claude", router: "router" };
function renderStatus(st) {
  // Claude is an optional cloud brain: without an API key it's "off", not an error, and every
  // question is answered on-device. Real failures of the on-device models still show red.
  const claudeOff = (st.claude || "").startsWith("error");
  $("model-status").innerHTML = Object.entries(st).map(([k, v]) => {
    const off = k === "claude" && claudeOff;
    const cls = off ? "off" : v.startsWith("error") ? "error" : v;
    const text = off ? "off (optional)" : v.startsWith("error") ? "error" : v;
    const tip = off ? "Optional cloud brain. Add ANTHROPIC_API_KEY to .env and restart to enable it." : v;
    const title = v.startsWith("error") ? ` title="${esc(tip)}"` : "";
    return `<span class="pill ${cls}"${title}>${STATUS_NAMES[k] || k}: ${text}</span>`;
  }).join("");
  const claudeOpt = $("brain-mode").querySelector('option[value="claude"]');
  claudeOpt.disabled = claudeOff;
  claudeOpt.textContent = claudeOff ? "Claude (cloud) – no key" : "Claude (cloud)";
  if (claudeOff && $("brain-mode").value === "claude") {
    $("brain-mode").value = "auto";
    sendJSON({ type: "brain", mode: "auto" });
  }
}

let lastCar = null;
function renderCar(c) {
  const set = (id, v) => { const el = $(id); if (el.innerHTML !== v) { el.innerHTML = v; if (lastCar) el.closest(".tile").classList.remove("flash"), void el.offsetWidth, el.closest(".tile").classList.add("flash"); } };
  set("car-temp", `${c.temperature_f}°F`);
  set("car-fan", `fan ${c.fan_level}`);
  set("car-media", c.media ? esc(c.media) : "off");
  set("car-vol", `vol ${c.volume}`);
  set("car-nav", c.destination ? `${esc(c.destination)} · ${c.eta_min} min` : "—");
  set("car-windows", Object.entries(c.windows).map(([k, v]) => `${k}: ${v}`).join("<br>"));
  set("car-seats", Object.entries(c.seat_heater).map(([k, v]) => `${k}: ${v}`).join("<br>"));
  set("car-misc", `defrost ${c.defrost ? "on" : "off"}<br>wipers ${c.wipers}`);
  lastCar = c;
}

// ------------------------------------------------------------------ microphone
let micReady = false, pttDown = false;
async function initMic() {
  if (micReady) return;
  const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 } });
  const ctx = new AudioContext({ sampleRate: 16000 });
  await ctx.audioWorklet.addModule("/static/mic-worklet.js");
  const node = new AudioWorkletNode(ctx, "mic-processor");
  node.port.onmessage = (e) => {
    if ($("listen-mode").value === "handsfree" || pttDown) sendBinary(MSG_AUDIO, e.data);
  };
  ctx.createMediaStreamSource(stream).connect(node);
  micReady = true;
}

async function pttStart() {
  if (pttDown) return;
  await initMic();
  speechSynthesis.cancel(); // barge-in: talking over the assistant stops it
  pttDown = true;
  $("mic").classList.add("live");
  sendJSON({ type: "ptt", down: true });
}
function pttEnd() {
  if (!pttDown) return;
  pttDown = false;
  $("mic").classList.remove("live");
  setTimeout(() => sendJSON({ type: "ptt", down: false }), 150); // let the last audio chunk arrive
}

$("mic").addEventListener("pointerdown", pttStart);
addEventListener("pointerup", pttEnd);
addEventListener("keydown", (e) => { if (e.code === "Space" && document.activeElement !== $("text") && !e.repeat) { e.preventDefault(); pttStart(); } });
addEventListener("keyup", (e) => { if (e.code === "Space" && document.activeElement !== $("text")) pttEnd(); });

$("ask").onsubmit = (e) => {
  e.preventDefault();
  const t = $("text").value.trim();
  if (!t) return;
  addMsg("user", esc(t));
  sendJSON({ type: "query", text: t });
  $("text").value = "";
};
$("brain-mode").onchange = (e) => sendJSON({ type: "brain", mode: e.target.value });
$("listen-mode").onchange = async (e) => { sendJSON({ type: "listen_mode", mode: e.target.value }); if (e.target.value === "handsfree") await initMic(); };
$("speed").oninput = (e) => { $("speed-val").textContent = `${e.target.value} mph`; sendJSON({ type: "speed", mph: +e.target.value }); };
$("recal").onclick = () => { earHist.length = 0; sendJSON({ type: "reset", which: "cabin" }); };

document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible")
    for (const s of Object.values(sources)) if (s.video.paused && (s.video.src || s.video.srcObject)) s.video.play().catch(() => {});
});

connect();
listCameras();
