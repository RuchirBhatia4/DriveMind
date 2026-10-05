# DriveMind: The ML Concepts, Explained

This is your study guide for the project. Each section covers **what** the technique is, **why** it's here, the **key math**, where to find it in the code, an **interview question** you should be able to answer, and an **exercise** to deepen it.

Read it alongside the code: every module's docstring is a short version of its section here.

```
 Browser (web/)                        Server (drivemind/)
 ─────────────                         ───────────────────
 road camera / dashcam ──JPEG 15fps──▶ perception/road.py     YOLO11 → ByteTrack → TTC ─┐
 cabin webcam ───────────JPEG 15fps──▶ perception/driver.py   landmarks → EAR/PERCLOS/pose ┤
 mic (AudioWorklet 16k) ─PCM─────────▶ audio/vad.py → asr.py  Silero VAD → Whisper turbo   │
                                       memory/scene_memory.py CLIP keyframes + retrieval   │
                                       brain/safety.py        workload + alert arbitration ◀┘
                                       brain/router.py ──▶ local_brain.py (Qwen2.5-VL 4-bit, MLX)
                                                      └──▶ claude_brain.py (Claude + tools)
 dashboard ◀──────── JSON results, alerts, replies ── server/session.py (threads + backpressure)
```

---

## 1. Real-time object detection (YOLO11)
**What.** A single-stage detector: one CNN forward pass predicts, for a grid over the image, box coordinates, an "objectness" score and class probabilities. Non-Maximum Suppression (NMS) then removes duplicate boxes (keep the highest-scoring box; drop others with IoU > threshold).

**Why here.** It runs in ~10–15 ms per frame on Apple Silicon (MPS). Two-stage detectors (Faster R-CNN) are more accurate on small objects but far too slow for a 15 fps loop.

**Key ideas.** IoU = area(A∩B)/area(A∪B). Detection quality is measured with mAP (mean average precision over IoU thresholds 0.5:0.95). YOLO11n ("nano") has ~2.6M parameters; the s/m/l/x variants trade speed for accuracy.

**Code.** `perception/road.py → RoadPerception.process`

**Interview Q.** *Why does a detector need NMS, and what goes wrong in crowded scenes?* (Overlapping pedestrians get suppressed as "duplicates". Soft-NMS and NMS-free designs such as YOLOv10/DETR address this.)

**Exercise.** Swap `yolo11n.pt` for `yolo11s.pt` and plot latency vs. how many distant cars you detect on a highway clip.

### 1b. Traffic-light state: classic CV on a detected crop
YOLO finds traffic lights but not their colour, so `classify_traffic_light` reads the lit lamp from the box. Not everything needs a neural net: a cheap, explainable rule on a crop that a network already localised is a common production pattern. The interesting part is how it failed.

- **Counting coloured pixels fails on real streets.** New York's housings are painted yellow, so side views read as "yellow". Green-painted housings read as "green". At 640 px a lit red lamp looks pale orange, the same hue as sunlit paint. Result: 36 of 85 lit lights wrong.
- **Structure beats colour.** A signal facing you has two dark, unlit lamps and one that isn't dark. Its position gives the state (top red, middle amber, bottom green), whatever colour the camera renders. Side views have no dark discs, and square boxes are pedestrian signals.
- **Unmistakable colour first, but colour lies too.** Pure red and strong green are trusted, with exceptions. A cyan sky sits in the green hue range and made red lights read **green**, so a lamp must be one compact blob inside the box. A camera that saturates at night renders red lamps with yellow-white cores, so at night position decides red vs amber.
- **Abstain rather than guess.** When the pattern isn't clear it returns None, and the assistant says it can't make out the light. That trades coverage (15-40% no answer) for few wrong colours.
- **Pre-registration earns its keep.** Each version was frozen with written predictions before a held-out test. The first two tests *failed* (LISA: night reds called yellow; 20 new BDD100K clips: 19 red lights called green). Both failures came from conditions missing from the design data, and both were fixed and retested on fresh data: 1 wrong of 50 lit lights, 0 of 26 false colours. Without held-out tests, the in-sample "0 of 85 wrong" would have shipped with a red-reads-green bug. See `eval/results/traffic_lights_preregistration.md`.
- **The VLM is no shortcut here.** Asked about zoomed crops, the 3B vision model named the housing's paint colour: 22 of 85 lit lights wrong, and a colour for 43 of 47 side views.

**Interview Q.** *Your colour classifier scores 100% on your dev set. What do you do before shipping?* (Test on data from other cities, cameras, times of day and weather that you never looked at, decided in advance, and expect it to fail. Then check which errors are dangerous (red → green) rather than counting all errors equally.)

## 2. Multi-object tracking (ByteTrack)
**What.** Assigns persistent IDs across frames. Each track has a **Kalman filter** (a constant-velocity motion model) that predicts where the box will be next frame. New detections are matched to predictions using IoU with the **Hungarian algorithm** (optimal bipartite matching; the `lap` package).

**ByteTrack's trick.** It does a second matching round using *low-confidence* detections, which rescues objects that are briefly blurred or occluded instead of dropping them and spawning a new ID.

**Why here.** Time-to-collision needs the *same* car's size over time. Without IDs you can't compute a rate of change.

**Interview Q.** *What is an ID switch, and how would you reduce them?* (Add appearance embeddings, as in DeepSORT/BoT-SORT, or tune the match thresholds.)

## 3. Monocular time-to-collision ("looming")
**What.** Estimate seconds-until-impact from **one camera**, with no depth.

**Math.** With a pinhole camera, an object of width W at distance Z appears with size s = fW/Z. If it approaches at constant speed v: TTC = Z/v = s/(ds/dt) = 1 / (d ln s / dt). Distance and speed cancel out. We fit a least-squares line to ln(s) over a 0.8 s window (robust to per-frame box jitter) and take 1/slope.

**Subtle bug we found and fixed.** A regression over a window estimates the slope at the window's *centroid*, which is 0.4 s in the past. Under constant closing speed TTC drops 1 s per second, so we subtract (t_now − t_centroid). Without the fix, warnings were ~0.4 s late. `tests/test_core.py` checks this against analytic ground truth.

**Ego corridor.** Only objects roughly ahead of us (center 40% of the frame, lower part) count as threats.

**What real footage taught us.** All of the above was validated on synthetic approaches, which have no ego-motion. On 16 minutes of real city driving (BDD100K) the original fixed "corridor" box raised "Brake!" 23% of the time. Fixes, each measured on the same footage (`eval/fcw_footage_eval.py`):
- **A distance-invariant in-path test.** An object at lateral offset X and width W appears at image offset fX/Z with width fW/Z, so offset ÷ width = X/W stays constant as you approach. Parked cars sit several widths off your heading for their whole approach; a car in your lane stays under ~0.7.
- **Confirmation.** Require 3 consecutive below-threshold estimates. This removes one-frame spikes, at the cost of ~0.3 s of warning time.
- **Outlier guard.** A real object can't grow by more than ~28% between frames. Jumps like that mean the tracker swapped identities (a scene cut, an occlusion), so restart that track's history.
- **Clipped boxes.** A tall vehicle close ahead gets cut off at the top of the frame, and a clipped height stops growing, which made TTC drift *up*. Remember each track's height/width ratio while it's fully visible and fill in the clipped side.

Result: 8.8% instead of 22.9%. That's still too high. Most remaining alarms happen because the rule assumes the image centre is your heading, so the next step is path estimation (lane detection). We also had to make the end-to-end test realistic: the old "zoom into a photo" approach overflowed the frame by TTC ~1.6 s and no longer looked like a real approach.

**Interview Q.** *When does looming TTC fail?* (Turning objects, partial occlusion that changes box size, ego-motion such as pitch over bumps, and objects that are moving away but still growing in the box because the crop is changing.)

**Exercise.** Read speed via OBD-II and fuse it in. With known ego-speed you can also estimate *distance*.

## 4. Driver monitoring: landmarks, EAR, PERCLOS, head pose
- **Face Landmarker** (MediaPipe): a BlazeFace detector, then a regression network for 478 3-D landmarks. Runs on CPU in ~4 ms.
- **EAR** (Eye Aspect Ratio) = (‖p2−p6‖ + ‖p3−p5‖) / (2‖p1−p4‖). It is scale-invariant and drops toward 0 when the eye closes.
- **Calibration.** We take the 80th percentile of EAR over the first ~3 s as this driver's "open" baseline, and use 72% of it as the threshold. Fixed thresholds fail across face shapes and camera angles.
- **Smoothing.** An EMA (exponential moving average, a 1-pole IIR low-pass filter) removes landmark jitter.
- **PERCLOS.** The fraction of the last N seconds with eyes closed. A widely used fatigue measure from NHTSA-funded research. Alarm thresholds vary between studies and products; we use 15% over 20 s, a choice that still needs validating on drowsy-driver data.
- **Head pose via PnP.** Given 6 2-D landmarks and a generic 3-D face model, `solvePnP` finds the rotation and translation that best project the 3-D points onto the 2-D ones. Rodrigues converts that to a rotation matrix, then to yaw, pitch and roll.
- **Hysteresis state machine.** The driver enters "drowsy" at PERCLOS > 15% and leaves at < 8%. Microsleep means eyes closed ≥ 1 s. Distracted means the head is turned or pitched down for ≥ 2 s.

**Code.** `perception/driver.py`

**Interview Q.** *Why do production DMS use infrared cameras?* (They work at night, see through sunglasses, and give consistent illumination. The 940 nm IR is invisible to the driver.)

### 4b. What evaluation taught us (Eyeblink8, our first real dataset)
We evaluated eye-closure detection on **Eyeblink8**: 8 videos, 4 people, 71,748 frames, 408 hand-labelled blinks, with per-frame "eye fully closed" flags. Full table: `eval/results/eyeblink8.md`.

- **Leave-one-person-out cross-validation.** With only 4 people, a random frame split would put the same person's frames in both train and test, and the model would learn *that person's eyes*. Holding out a whole person measures what matters: performance on a driver it has never seen. Thresholds were picked with an *inner* leave-one-person-out on the 3 training people, so the test person influences nothing (nested cross-validation).
- **The rule that "looked fine" wasn't.** The original calibrated-EAR rule scored blink F1 0.91, which sounds good. But its frame-level closed-eye F1 was only 0.50: it called many half-closed or looking-down frames "closed". PERCLOS, which drives the alerts, depends on frame-level accuracy, so it ran 4.5 points too high, and the system would have raised **~33 false microsleep alerts per hour on wide-awake people**. *Lesson: evaluate the metric your product actually consumes, not the one that's easiest to report.*
- **Temporal windows + feature fusion.** A gradient-boosted classifier over a 13-frame window (EAR normalized two ways, MediaPipe's eyeBlink blendshape, head pitch) reached blink F1 **0.949**, PERCLOS error **0.56 points**, and **0** false microsleeps per hour (held-out people).
- **Robustness beats a small F1 gain.** Logistic regression on EAR windows scored almost as well (F1 0.940), but its false-alarm rate swung from 0 to 16.6 per hour depending on the threshold. The boosted trees stayed at F1 0.946–0.951 with zero false microsleeps for every threshold from 0.1 to 0.9.
- **A strong free baseline.** MediaPipe's own eyeBlink score, with no training, already fixed the false alarms (PERCLOS error 0.41, the best of all), but missed partial blinks (recall 0.855). It's now the fallback when the trained model file is absent.
- **Training/serving skew.** The live `DriverMonitor` and the offline evaluation build features with the *same* function (`perception/eye_state.py: window_features`), and `eval/verify_online.py` checks the live system reproduces offline decisions frame for frame (100% agreement). Models that behave differently in production than in evaluation are one of the most common real-world ML failures.

**Limits.** 4 people, 40 minutes, indoors, awake people at a desk. This validates **eye-closure detection**, not drowsiness detection, which needs drowsy subjects (UTA-RLDD, NTHU-DDD, DMD).

### 4c. Real drowsiness: UTA-RLDD (out-of-domain)
6 people filmed themselves alert, low-vigilant and drowsy (self-reported, KSS). Full tables: `eval/results/rldd_fold3_part2.md`.
- **Out-of-domain generalization.** The eye model trained on Eyeblink8 (4 alert people, crisp blinks) separated drowsy from alert *worse* than MediaPipe's untrained blink score (long closures/min AUC 0.94 vs 1.00). Drowsy eyes droop and close slowly, a pattern absent from the training data. A model can win on its own benchmark and lose on the task you care about. Fix: train on data that contains the target behavior.
- **Confounds.** Recordings came from different phones at 12–30 fps, and drowsy videos happened to have higher frame rates. We re-ran everything with every video degraded to 12 fps (a *control*); conclusions held. Always ask what *else* differs between your classes.
- **Error bars for small n.** 13/18 videos correct is 72%, but the 95% Wilson interval is 49–88%. That overlaps the paper's 70% and humans' 60%, so the honest claim is "on par".
- **Within-person vs. between-person.** Per video, PERCLOS separates states well (AUC 0.83). But the app's microsleep alerts rose with drowsiness for only 3/6 individuals: some drowsy people barely close their eyes. Population metrics hide per-user failures.
- **Label noise.** Labels are self-reported states for whole 10-minute videos, so a "drowsy" video contains alert minutes. That caps achievable window-level accuracy.

### 4d. Pre-registration and a true held-out test
With 6 people, *any* rule you tune will look good on them. So: choose on development people, **write the choice down and commit it**, then test once on new people (RLDD fold 5). We also pre-committed the *ship criterion*. See `eval/drowsy_rule.py`.
- **What happened.** The new long-closure rule tied the old PERCLOS rule on detection (5/6 drowsy videos) with zero false alarms. That's a modest, honest win: fewer false alarms, not better detection. Without the held-out test we'd have claimed more.
- **Optimism of small-sample cross-validation.** The window classifier scored 72% in leave-one-person-out cross-validation on 6 people but 56% on 6 new people. Each fold trained on only 5 people, and all our design choices were made on those same 6. Reported dev numbers are an upper bound.
- **Transparency about changes.** The first selection run picked a rule that nagged awake people 4.5 times an hour (caught by the Eyeblink8 check). We added that constraint *before* touching test data and recorded the change in the pre-registration. Changing the plan is fine; hiding that you changed it is not.
- **Don't drop inconvenient test data.** One test video was 288×162 pixels with a face visible only 72% of the time. No exclusion rule was pre-registered, so it stayed in.
- **Sensitivity vs. false alarms.** The exploratory "any closure ≥ 0.7 s" rule caught 6/6 drowsy people but cost ~2 false alarms per hour on alert drivers. Choosing between them is a product decision (how much nagging buys how much safety), and the data's job is to make the trade-off visible.

- **Replication.** A second pre-registered test on 6 more people didn't just repeat the tie: the shipped rule beat the old one (5/6 vs 4/6 detected, 1.0 vs 4.1 false alerts/h). Pooled over 12 unseen people it's 10/12 vs 9/12 detected, with 1 vs 4 false alerts. One test can mislead in either direction; agreement across independent tests is what builds confidence. Even so, the 95% intervals (55–95% vs 47–91%) still overlap, so the honest claim is "consistently ahead", not "proven better".
- **Learning curves.** The same classifier trained on 6 vs 12 people scored 44% vs 56% on the new people (AUC 0.83 vs 0.89). More data helped, which says the model is data-starved rather than badly designed, and that the 72% cross-validation figure on 6 people was optimism.
- **Operational lesson.** Google Drive caps downloads per file per day, and our quick "is it available?" probe started a real download that used up the last one. The fix was a mirror with per-video files, verified byte-for-byte against the official zips before trusting it.

**Exercise.** Add a non-eye cue (yawning, head nodding, or slower blink *re-opening* speed) and test whether it catches the drowsy drivers who barely close their eyes, using the same pre-registration routine.

## 5. Streaming voice activity detection (Silero VAD)
A ~2 MB recurrent network scores each 32 ms chunk for speech probability, carrying state between chunks (so each chunk is O(1)). We add **hysteresis** (start above 0.5, end below 0.35), a **hangover** (600 ms of silence ends the utterance) and a **pre-roll** (300 ms before onset, so the first syllable isn't clipped).

**Why it matters.** Whisper hallucinates on silence ("Thanks for watching!"). The VAD also sets *when we can start answering* (endpointing latency), which users feel directly.

## 6. On-device speech recognition (Whisper large-v3-turbo on MLX)
An encoder–decoder transformer over log-Mel spectrograms. *Turbo* prunes the decoder from 32 layers to 4: the encoder runs once (parallel), while the decoder runs per token (sequential). Cutting decoder depth gives ~8× faster decoding for a small loss in word error rate (WER).

We bias the vocabulary with `initial_prompt` (domain terms) and drop segments with high `no_speech_prob`.

**Measured:** ~0.5–0.6 s for a 3–4 s utterance on this Mac.

## 7. Vision-language models + 4-bit quantization (Qwen2.5-VL-3B on MLX)
- **Architecture.** A ViT encodes the image into patches (14×14 px, merged 2×2). A projector maps those patches into the LLM's embedding space, and the LLM attends over the image and text tokens together.
- **Image tokens scale with area.** At 448 px wide a frame is ~250 visual tokens; at 896 px it is ~1000. We downscale for normal queries and *upscale* retrieved frames when the user wants text read ("what did that sign say?").
- **Quantization.** Group-wise affine 4-bit: each block of 64 weights stores a scale and bias, with w ≈ scale·q + bias, q ∈ {0..15}. This takes 3B params from ~6 GB to ~2 GB.
- **Prefill vs. decode.** Prefill is compute-bound (we measure ~370–690 tok/s on the prompt). Decode is memory-bandwidth-bound (~110 tok/s). Quantization helps decode the most, because each generated token has to stream all the weights.
- **Prompted tool calling + few-shot.** The 3B model initially made only one of two requested tool calls. Adding worked examples to the prompt fixed it (*in-context learning*).
- **The catch: small models copy examples.** With a visual example in that same prompt, "Is the light green?" got "Yes, the light ahead is green." word for word, *at a red light*. Removing the examples made it call tools for plain questions instead ("Navigating to the red light"). The fix was to stop sharing one prompt: only car commands see the tool prompt and its examples, and everything else gets a plain "look and answer" prompt (section 8).
- **Small details are the limit.** At 448 px wide a distant traffic light is a few pixels. On one intersection frame the model said "The light is red." while both signals facing us were green (a lit orange "don't walk" hand was bigger). Light colour now comes from a dedicated classifier (section 1b), and the model's own light claims are replaced with its reading. The model still mixes up details: "How many people can you see?" got "I can see 7 cars".

**Code.** `brain/local_brain.py`

**Interview Q.** *Your on-device LLM is too slow. List five levers.* (Smaller or quantized model; fewer image tokens; prefix/KV caching of the static system prompt; speculative decoding; shorter outputs; streaming TTS so speech starts at the first sentence.)

## 8. Tool calling & the safety envelope
The LLM proposes actions as structured JSON; deterministic code validates, clamps or blocks them (no 200°F, no fully-open windows above 55 mph). When the envelope overrides the model, **the envelope's message is what gets spoken**. We found the local model confidently saying "Opening all the windows" while the car actually vented them, and fixed it.

This is the core idea behind using LLMs in safety-relevant products: **the LLM is advisory, the guardrails are code.** (ISO 26262 functional safety, ISO 21448 SOTIF.)

**Routing is code too.** A few regular-expression rules (`brain/common.py`) decide which path a request takes, instead of letting the 3B model choose:

| Request | Route | What the model may do |
|---|---|---|
| About the past ("what did that sign say?") | visual memory | answer from the best-matching past frame (section 9) |
| Traffic-light colour ("is the light green?") | light classifier | nothing: the classifier answers in any brain mode (section 1b) |
| Go/no-go ("can I go now?") | describe only | describe the scene; code drops any sentence claiming it's safe, puts the classifier's light reading first, and adds "I can't judge that for you, so please check for yourself." |
| Car command ("I'm cold, play some jazz") | tools | the JSON tool prompt, without the memory tool |
| Anything else ("what kind of street is this?") | look and answer | a plain prompt with the live image and no tools; any light colour it claims is replaced with the classifier's |

Before this, the model told the driver "it's safe to go", spoke raw JSON aloud, and called tools for plain questions. On real BDD100K frames (`eval/brain_qa_eval.py`), the routed version got 9/9 light-colour questions right on a red frame, a green frame and a frame with tiny lights. It never claimed it was safe (6/6), and made the right tool calls for 3/3 commands. A bug worth knowing about: the first check for "the model claimed a light colour" matched "the car with the red tail lights" and replaced a car answer with the light's colour. It now needs the colour next to "light" or "signal", excluding tail, brake and head lights.

## 9. Semantic visual memory (CLIP + retrieval)
**CLIP** is trained contrastively (InfoNCE loss) so that matching image–caption pairs have high cosine similarity and mismatched pairs low. Result: text and images share one embedding space.

- **Keyframe selection.** We store a frame when 1 − cos(e_t, e_last) > 0.10 (the *meaning* changed) or 5 s have passed. Pixel differencing would fire on every lighting change.
- **Retrieval.** score = cos(text_query, frame) + 0.05·recency. The top frame goes back to the VLM as context. That's RAG, with frames instead of documents.
- **Bounded memory.** A 3-minute ring buffer.
- **When to search is a rule, not the model's choice.** In a recorded demo the 3B model answered "Which car is on my left?" from a frame 17 s old. Its choice also varied: in the eval run before the fix it left memory alone for present-tense questions but skipped it for 2 of 6 questions about the past, and one of those misses spoke raw JSON aloud. Now a question searches memory only when its wording points to the past (*did, was, passed, ago, the last sign*…) and to something visible (*sign, car, exit, colour*…). The Claude brain can still ask for `recall_scene`, but the server refuses it for present-tense questions. Result: 0/8 present-tense questions searched memory and 6/6 past-tense ones did (4/6 before).

**Interview Q.** *CLIP retrieval returns the wrong frame for "the exit sign". How would you improve it?* (Region-level embeddings for crops of detected signs, OCR text indexed alongside, a larger CLIP model, or re-ranking the top-k with the VLM.)

## 10. Hybrid edge/cloud routing
A transparent heuristic: complex or long queries go to Claude, simple ones stay local. A **circuit breaker** marks the cloud as down for 30 s after a failure, and the system **degrades gracefully** to on-device.

**Next step (a great exercise).** Make it a *learned router*. Log queries plus whether the local answer was right (judged by Claude or by you), then train a small classifier on sentence embeddings to predict "local will fail".

## 11. Attention management (when to speak)
Decision-level sensor fusion: workload = f(TTC, driver state, vulnerable road users in path).

- At **critical** workload, replies are *deferred* until the hazard clears.
- At **high** workload, replies are shortened to one sentence.
- Alerts are arbitrated by priority and rate-limited with cooldowns, because alert fatigue makes drivers switch systems off.
- Drowsiness triggers **conversation** rather than a chime.

## 12. Real-time systems engineering
- **Pipeline parallelism.** One worker thread per stage, so a 2 s VLM answer never delays a collision warning.
- **Backpressure (latest-frame-wins).** If a stage is busy, the new frame is dropped rather than queued. A queue in a real-time system just turns into latency. Watch the "dropped" counter: when the VLM is answering, YOLO and the VLM contend for the same GPU and road FPS dips. On a car you'd pin perception to a dedicated accelerator (DLA/NPU).
- **Client backpressure.** The browser skips frames when `ws.bufferedAmount` grows.
- **Binary protocol.** Raw JPEG/PCM with a 1-byte type header (base64 would add 33% overhead).
- **Deterministic resource cleanup (a real bug we hit).** After a camera session ended, the road stage of the *next* session froze forever. A stack dump (`kill -USR1 <pid>`, via `faulthandler`) showed why: Python's garbage collector ran MediaPipe's `__del__` → `close()` *inside the road thread*, in the middle of PyTorch loading YOLO weights, and `close()` waited on an internal thread that never answered. Fix: close native resources explicitly, on their own thread, when a session ends, and never leave it to the garbage collector.
- **Watchdogs.** Backpressure had made the freeze *silent*: frames were just "dropped". Now a perception stage busy for more than 5 s is reported and replaced with a fresh worker. Safety-relevant pipelines need to detect their own stalls.

---

## Measured numbers (MacBook, Apple Silicon, 24 GB)
| Stage | Latency |
|---|---|
| YOLO11n + ByteTrack + TTC | 9–18 ms / frame |
| Face landmarks + eye-state model + PERCLOS/pose | ~6.6 ms median, 9 ms p95 / frame |
| CLIP keyframe embedding | ~18 ms |
| Whisper turbo (3 s utterance) | ~0.5–0.6 s |
| Qwen2.5-VL-3B 4-bit answer | ~1.6–2.2 s (prefill ~1000 tokens, decode ~110 tok/s) |
| Speech end → answer (local) | ~2.5 s |

## Where to go next (in order of interview value)
1. **Benchmark on real datasets** (DMD / NTHU-DDD for drowsiness; DriveLM or BDD100K for road VQA) and publish a table.
2. **Cut voice latency.** Prefix-cache the system prompt, stream TTS by sentence, and try speculative decoding.
3. **Learned router** (section 10).
4. **Lane detection** (e.g. UFLD) → lane-departure warning + lane-relative ego corridor.
5. **Port perception to Jetson Orin / TensorRT** and report FPS and power.
6. **ROS 2 node wrapper** so it plugs into robotics and AV stacks.
