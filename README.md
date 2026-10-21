# DriveMind: Multimodal In-Vehicle AI Assistant

**An on-device co-pilot that sees the road, watches the driver, listens to the cabin, and reasons across all three in real time.**

<p align="center">
  <img src="docs/media/demo.webp" width="100%" alt="DriveMind running live: tracking cars at a Manhattan crosswalk; a collision warning and a microsleep alert at the same moment; a distraction alert">
</p>

<sub>Recorded live on a MacBook Pro; everything runs on-device. Road footage: <a href="https://bdd-data.berkeley.edu/">BDD100K</a> (Berkeley DeepDrive), © 2018 The Regents of the University of California, used for non-commercial, educational purposes under the <a href="https://github.com/bdd100k/bdd100k/blob/master/doc/source/license.rst">BDD100K license</a>.</sub>

DriveMind fuses three systems that today's cars keep separate:

- **ADAS-style road perception:** object detection, multi-object tracking, and monocular time-to-collision for forward-collision warnings.
- **Driver monitoring:** eye aspect ratio, PERCLOS, yawns, and head pose, used to detect drowsiness, microsleeps, and distraction.
- **A voice assistant with eyes:** streaming VAD → Whisper → a 4-bit vision-language model with tool calling, plus a CLIP-indexed visual memory so you can ask *"what did that sign say?"* after you've passed it.

<p align="center"><img src="docs/media/voice_qa.png" width="85%" alt="Spoken question 'Is that light green?' answered on-device: Whisper 0.6 s, VLM 1.7 s, 2.3 s end to end"></p>

A safety layer decides *when* the assistant may speak. It defers answers during hazards, shortens them under high workload, and starts a conversation when the driver gets drowsy.

Everything runs **on-device** on Apple Silicon (MLX + MPS). An optional **Claude** cloud brain handles harder questions, with automatic fallback to local when the network drops.

## Architecture

```
 Browser dashboard                         Python server (FastAPI + WebSocket)
 ─────────────────                         ────────────────────────────────────
 road cam / dashcam video ─JPEG─▶ YOLO11n → ByteTrack → looming TTC ───────┐
 cabin webcam ─────────────JPEG─▶ MediaPipe landmarks → EAR · PERCLOS · PnP ┤──▶ attention manager ──▶ alerts
 microphone (16 kHz PCM) ────────▶ Silero VAD → Whisper large-v3-turbo      │     (workload gating)
                                  CLIP keyframe memory (3 min) ◀────────────┘
                                  router ─▶ Qwen2.5-VL-3B 4-bit (MLX, on-device)
                                        └▶ Claude (vision + tool use, optional)
                                  vehicle tools behind a deterministic safety envelope
```

Each stage runs on its own worker thread with **latest-frame-wins backpressure**, so a slow LLM answer can never delay a collision warning.

## Measured performance (MacBook, Apple Silicon, 24 GB)

| Stage | Latency |
|---|---|
| Road: YOLO11n + ByteTrack + TTC | 9–18 ms / frame |
| Driver: landmarks + trained eye-state model + PERCLOS/head pose | ~6.6 ms median / frame |
| Visual memory: CLIP keyframe embedding | ~18 ms |
| Speech recognition: Whisper turbo, 3 s utterance | ~0.5–0.6 s |
| On-device VLM answer (Qwen2.5-VL-3B, 4-bit) | ~1.6–2.2 s, decode ~110 tok/s |
| End of speech → spoken answer (fully local) | ~2.5 s |
| TTC accuracy vs. ground truth | within 0.15 s (unit test); simulated approach converges to within 0.05 s |

## Evaluation: driver eye-closure detection (Eyeblink8)

The driver monitor's eye-closure detector was evaluated on the public
[Eyeblink8](https://www.blinkingmatters.com/research) dataset (Fogelton & Benesova, 2016):
8 videos, 4 people, 71,748 frames, 408 hand-labelled blinks. Every number below comes from
**leave-one-person-out** cross-validation, so each person is scored by a model that never saw them.

| Method | Blink F1 | Closed-frame F1 | PERCLOS error (pts) | False microsleep alerts / h |
|---|---|---|---|---|
| Textbook rule (EAR < 0.2) | 0.886 | 0.416 | 7.91 | 57.2 |
| Original calibrated-EAR rule | 0.907 | 0.497 | 4.48 | 33.1 |
| MediaPipe eyeBlink score (untrained fallback) | 0.911 | 0.843 | 0.41 | 0.0 |
| **Gradient-boosted trees, 13-frame window (trained, shipped)** | **0.949** | **0.867** | **0.56** | **0.0** |

The original rule's blink score looked fine, but it would have raised ~33 false microsleep
alerts per hour on people who were wide awake. The trained model is robust to its
threshold (F1 0.946–0.951, zero false microsleeps, for every threshold from 0.1 to 0.9), and
the live system reproduces the offline decisions frame-for-frame (`eval/verify_online.py`).
Full results, per-person breakdown and protocol: [eval/results/eyeblink8.md](eval/results/eyeblink8.md).

**Limits:** 4 people, 40 minutes of awake subjects at a desk. This measures eye-closure
detection, not drowsiness itself, which still needs a drowsy-driver dataset (UTA-RLDD, NTHU-DDD).

Reproduce (downloads 313 MB; the dataset and trained weights are GPL-3 derived, so they're git-ignored):

```bash
mkdir -p data && curl -L -o data/eyeblink8.zip https://www.blinkingmatters.com/files/upload/research/eyeblink8.zip
(cd data && unzip -q eyeblink8.zip && rm eyeblink8.zip)
.venv/bin/python -m eval.eyeblink8_extract   # MediaPipe features for 71k frames (~5 min)
.venv/bin/python -m eval.eyeblink8_eval      # LOPO evaluation + trains models/eye_state_gbm.joblib
.venv/bin/python -m eval.verify_online       # live DriverMonitor == offline pipeline
```

Without `models/eye_state_gbm.joblib`, the driver monitor falls back to MediaPipe's eyeBlink score.

## Evaluation: real drowsiness (UTA-RLDD)

Eyeblink8 only had awake people. To test actual drowsiness I used part of
[UTA-RLDD](https://sites.google.com/view/utarldd/home) (Ghoddoosian et al., CVPRW 2019):
6 participants (fold 3, part 2) who filmed themselves on their own phones/webcams for ~10 min
each while alert, low-vigilant and drowsy (self-reported on the Karolinska Sleepiness Scale), 3.3 h in total.
**No RLDD data was used to train the eye model**, so this is an out-of-domain test.

**The shipped app, unchanged** (trained eye model + alert rules):

| Self-reported state | PERCLOS | Microsleep alerts / h | Drowsy alerts / h |
|---|---|---|---|
| Alert | 1.7% | 0.0 | 0.9 |
| Low vigilant | 4.7% | 15.8 | 3.0 |
| Drowsy | 9.0% | 65.8 | 16.4 |

Alerts rise with drowsiness and stay near zero for alert drivers. But per person, the app's
microsleep alerts fire more when drowsy for only **3 of 6** people. Some drowsy people barely close their eyes.

**Out-of-domain lesson:** MediaPipe's untrained eyeBlink score separated drowsy from alert better
(long eye closures per minute: ROC-AUC 1.00, higher when drowsy for 6/6 people) than the model
I trained on Eyeblink8 (AUC 0.94, 5/6). That model learned crisp blinks from four alert people;
drowsy eyes droop and close slowly.

**Trained drowsiness classifier** (1-minute windows of blink features, logistic regression,
leave-one-participant-out): **13/18 videos correct (72%, 95% CI 49–88%)**, alert-vs-drowsy
window ROC-AUC 0.95. On this fold the RLDD paper reports 70% for its HM-LSTM and 60% for human
judges, so we're on par. With 18 test videos the interval is wide, and this is not a claim of beating the paper.

**Controls:** frame rates differed between recordings (12–30 fps), so everything was re-run with
every video degraded to 12 fps; conclusions unchanged (classifier still 13/18, AUC 0.95).
Full tables: [eval/results/rldd_fold3_part2.md](eval/results/rldd_fold3_part2.md),
[12 fps control](eval/results/rldd_fold3_part2_fps12.md).

**Limits:** 6 people; labels are self-reported states for whole videos; people sat at home, not driving.

### Held-out test of a new alert rule (pre-registered)

The results above suggested alerting on *long* eye closures instead of PERCLOS. To avoid
fooling myself with 6 people, I chose the new rule using only those 6 development people (plus
Eyeblink8 as an awake-driver false-alarm check) and **committed it to git before processing
6 new people** (RLDD fold 5, part 1). See `eval/results/preregistration.json`, commits
`20c45c5`/`28e74f6`.

**Frozen rule:** alert when MediaPipe's eyeBlink score shows ≥ 4 eye closures of ≥ 0.5 s
within 60 s, or one closure ≥ 1 s.

| On 6 new people (3.4 h) | False alerts/h, alert videos | Drowsy videos alerted | People alerted more when drowsy |
|---|---|---|---|
| Previous rule (PERCLOS > 15% + microsleep) | 0.0 | 5/6 | 5/6 |
| **New rule (shipped)** | **0.0** | **5/6** | **5/6** |
| Exploratory: any closure ≥ 0.7 s | 2.0 | 6/6 | 6/6 |

- **Detection tied** on new people. The new rule's gain is false alarms: **zero** across all
  12 RLDD people and Eyeblink8's awake people, against one each for the previous rule. It met
  the pre-registered ship criterion, so it now drives the app's alerts
  (`drivemind/perception/drowsiness.py`). The live code makes the same alert/no-alert call
  as the offline rule on all 44 videos, at 30 fps and at the dashboard's 15 fps (`eval/verify_drowsiness_rule.py`).
- **Catching the last drowsy person costs ~2 false alarms per hour**, per the exploratory rule.
- **The drowsiness classifier did not generalize.** 72% in cross-validation on the 6 development people fell
  to **10/18 videos (56%, 95% CI 34–75%) on new people**, near human judges (58%) and below
  the paper's 65% (trained on 48 people). Not shipped.

Full table: [eval/results/rldd_heldout_test.md](eval/results/rldd_heldout_test.md).

### Second held-out test (pre-registered, 6 more unseen people)

Same frozen rules, no re-tuning, plan committed (`ff46dae`) before the data was downloaded
(RLDD fold 2, part 1, via a Kaggle mirror whose files match the official ones byte-for-byte).

| On 6 more new people (3.1 h) | False alerts/h, alert videos | Drowsy videos alerted | People alerted more when drowsy |
|---|---|---|---|
| Previous rule (PERCLOS > 15% + microsleep) | 4.1 | 4/6 | 3/6 |
| **Shipped rule** | **1.0** | **5/6** | **5/6** |
| Exploratory: any closure ≥ 0.7 s | 3.0 | 5/6 | 4/6 |

**Pooled over both held-out tests (12 unseen people, 6.4 h):**

| Rule | False alerts in 2.0 h of alert driving | Drowsy drivers caught (95% CI) |
|---|---|---|
| Previous rule | 4 | 9/12 (47–91%) |
| **Shipped rule** | **1** | **10/12 (55–95%)** |
| Exploratory | 5 | 11/12 (65–99%) |

The shipped rule is ahead on both detection and false alarms, but with 12 people the intervals
overlap heavily: consistent with an improvement, not proof of one. Two of the 12 drowsy drivers
barely closed their eyes at all, so no eye-closure rule caught them; that's the case for adding other cues.

**More training data helped the classifier, modestly:** trained on 6 people it scored 8/18 videos
(44%) on these new people; trained on 12 it scored 10/18 (56%, 95% CI 34–75%), window AUC 0.83 → 0.89.
That's still near human judges (58%) and below the paper's 65% (trained on 48 people), so it isn't shipped.
Results: [second test](eval/results/rldd_heldout_test_Fold2_part1.md), [pooled](eval/results/rldd_heldout_pooled.md).

```bash
.venv/bin/gdown -O data/rldd/Fold3_part2.zip 1LZU5KfJkFMj2pIkGwb3RjHDazoUfxSYQ   # 7.5 GB
.venv/bin/python -m eval.rldd_extract data/rldd/Fold3_part2.zip --workers 4      # ~10 min
.venv/bin/python -m eval.rldd_eval && .venv/bin/python -m eval.rldd_eval --cap-fps 12
# held-out test (fold 5 part 1, gdown id 16w30wEgcAbaUr4gx9o4sl_2TbIunD5yA, 10.9 GB)
.venv/bin/python -m eval.rldd_extract data/rldd/Fold5_part1.zip --workers 4
.venv/bin/python -m eval.drowsy_rule test          # uses the frozen preregistration.json
.venv/bin/python -m eval.verify_drowsiness_rule    # live rule == evaluated rule
# second held-out test: per-video download from the Kaggle mirror (needs a Kaggle token in ~/.kaggle/)
.venv/bin/python -m eval.rldd_extract --kaggle-part Fold2_part1 --workers 4
.venv/bin/python -m eval.drowsy_rule test --test-fold Fold2_part1 --extra-train Fold5_part1
.venv/bin/python -m eval.drowsy_rule pooled
```

## Evaluation: collision warnings on real driving (BDD100K)

The first collision-warning version was only tested on synthetic approaches. Run on 24 random
40-second clips of ordinary US city driving from [BDD100K](https://bdd-data.berkeley.edu/)
(16 min, no crashes), its red **"Brake!" alarm was on 23% of the time**: parked cars at the
curb fell inside its fixed "corridor" box and loomed as the car drove past. Fixes, each
measured on the same footage:

| Version | Critical ("Brake!") | Warning |
|---|---|---|
| Fixed corridor box (original) | 225 s (22.9%) | 181 s |
| In-path test using offset ÷ object width (constant with distance) + 3-frame confirmation | 84 s (8.6%) | 63 s |
| + ignore implausible box jumps (tracker identity swaps) + fill in clipped box sides | **86 s (8.8%)** | **64 s** |

A genuine approach still triggers it. On a constant-speed simulated approach, the estimate converges
to within 0.05 s of the truth and "Brake!" fires at a true time-to-collision of ~1.3–1.4 s. The 3-frame
confirmation costs about 0.3 s of warning time compared with the 1.6 s threshold.

**Still a limitation:** about 9% is far too high for a product. Most remaining alarms are parked
cars when the street curves or the camera isn't pointed straight ahead, because the rule assumes
the image centre is the car's heading. The real fix is estimating the car's path (lane
detection); see the roadmap. Results: [eval/results/fcw_bdd100k.md](eval/results/fcw_bdd100k.md);
reproduce with `.venv/bin/python -m eval.fcw_footage_eval` (needs a Kaggle token; downloads ~0.5 GB).

## Evaluation: traffic-light colour (BDD100K, LISA)

The on-device model said *"The light is red."* at an intersection where both signals facing the car
were green: at the 448 px it sees, they're a few pixels wide. Light colour now comes from a small
classic-CV classifier on the boxes YOLO already finds (`classify_traffic_light` in `road.py`). It
doesn't count coloured pixels, which is what the first version did: in New York that read yellow-painted
housings as "yellow" and orange-looking red lamps as "yellow", 36 of 85 lit lights wrong in the local
clips. Instead it finds the lit lamp: an unmistakable colour first (a compact green blob, pure red),
otherwise the one lamp position that isn't a dark disc (top red, middle amber, bottom green). It gives
no answer when unsure, because "I can't make out the light" beats a wrong colour.

Every version was frozen and its predictions written down before testing on lights it had never seen
([pre-registration and results](eval/results/traffic_lights_preregistration.md)). Two tests failed, and
each failure led to a fix and a new test:

| Held-out test (version tested) | Lit lights | Wrong colour, old → new | New: no answer | Colour given to side views / signs, old → new |
|---|---|---|---|---|
| LISA crops, labelled by its authors (first version) | 27,521 | 0.5% → **5.5%** ✗: red lights at night called yellow | 32% | (none in set) |
| 20 random BDD100K clips (revision 1) | 82 | 33% → **24%** ✗: 19 red lights called **green** under a cyan sky | 12% | 27% → 9% |
| 20 more random BDD100K clips (final) | 50 | 2% → **2%** (1 of 50) | 22% | 12% → **0%** |

The final version on the data used to design it: 0 of 85 wrong colours on the local clips, 0.4% on LISA,
1 of 82 on the first BDD100K test.

**Limits:**
- It gives no answer for 15-40% of lit lights depending on the footage (small, blurred or dull lamps).
- It missed the only amber light in the BDD100K clips. On LISA it reads 40% of amber lights.
- The BDD100K labels are by eye. They're in `eval/labels/`, with the script that rebuilds the labelling sheets.
- Frames 2 s apart are correlated, so 50 lights are not 50 independent scenes.

We also tried asking the 3B vision model about zoomed crops instead. It named the housing's paint colour:
22 of 85 lit lights wrong. Reproduce with `.venv/bin/python -m eval.traffic_light_eval dev|lisa|test|test2`.

## Evaluation: voice answers (visual memory, go/no-go questions, commands)

In a recorded demo, the on-device model answered *"Which car is on my left?"* from a visual-memory
frame 17 seconds old. Leaving such choices to a 3B model failed in other ways too: it copied an
example answer from its prompt (*"Yes, the light ahead is green."* at a red light), told the driver
"it's safe to go", and sometimes spoke raw JSON aloud. Now simple rules pick the route:

- visual memory only for questions about the past
- the light classifier above for traffic-light colour, in any brain mode
- a describe-only prompt for go/no-go questions, where the light's colour also comes from the classifier
- the tool prompt only for car commands
- a plain "look and answer" prompt for everything else

A small check of 32 questions on real BDD100K frames:

| Check | Result |
|---|---|
| Questions about the present that searched memory | **0/8** |
| Questions about the past that searched memory | **6/6** (4/6 before) |
| Light colour on a red frame, a green frame, and the small-lights frame above | **9/9** |
| "Is it safe to go?" / "Can I go now?" answered without claiming it's safe | **6/6** |
| Car commands with the right tool calls | **3/3** |

**Still a limitation:** the 3B model misses or mixes up details. It answered *"How many people can you
see?"* with *"I can see 7 cars"*, read a "KEEP INTERSECTION CLEAR" sign as "KEEP HIGHER CLEAR", and
answers *"Which car is on my left?"* vaguely. Results:
[eval/results/brain_qa_after.json](eval/results/brain_qa_after.json); reproduce with
`.venv/bin/python -m eval.brain_qa_eval` (needs a Kaggle token).

## Quick start

**Requirements:** a Mac with Apple Silicon (M1 or newer; the speech and vision-language models
run on Apple's MLX framework), Python 3.11 (tested), about 9 GB of free disk (5 GB of models plus
packages), Chrome, and a webcam + microphone. `ffmpeg` (`brew install ffmpeg`) is only needed for
`scripts/e2e_check.py`.

```bash
git clone https://github.com/RuchirBhatia4/DriveMind.git
cd DriveMind
scripts/setup.sh      # one time: venv, dependencies, ~5 GB of model weights (~3 min with a fast connection)
scripts/run.sh        # every time; runs offline and opens the dashboard in Chrome when ready (~15-30 s)
```

The pills at the top of the dashboard turn green as each model finishes loading.
The dashboard (**http://127.0.0.1:8000**) opens by itself; allow camera and microphone access, then:

1. **Driver monitor:** your webcam is picked automatically. Look ahead for ~3 s while it calibrates to your eyes.
2. **Road camera:** click **Load dashcam video** and choose a driving clip. No dashcam? `.venv/bin/python scripts/make_demo_reel.py`
   (needs a Kaggle token) builds a 78-second reel of real BDD100K city driving at `data/demo/drivemind_demo_reel.mp4`,
   with a `CREDITS.txt` to include wherever you publish video made from it. Or pick a second camera.
3. **Talk:** hold **Space** (or the mic button) and ask:
   - "How many cars are ahead of us?" / "Is the light green?" (answered by the light classifier, instantly)
   - "I'm freezing, warm it up and play some jazz" (makes two tool calls)
   - "What did that sign say?" (searches visual memory)
   - "Open all the windows" with the speed slider above 55 mph (the safety envelope vents them instead)
4. Close your eyes for 2 s, or look away from the screen, to trigger the driver alerts.

**Optional:** put `ANTHROPIC_API_KEY=...` in `.env` to enable the Claude brain. *Auto* mode then routes complex questions to the cloud and falls back to on-device if it's unreachable.

## Project layout

```
drivemind/
  perception/road.py      detection, tracking, time-to-collision, traffic-light color
  perception/driver.py    EAR, PERCLOS, yawns, head pose (PnP), hysteresis state machine
  perception/eye_state.py trained eye-closure model (streaming; same features as eval/)
  audio/vad.py            streaming Silero VAD with hysteresis, hangover, pre-roll
  audio/asr.py            Whisper turbo on MLX with domain biasing + hallucination guards
  memory/scene_memory.py  CLIP keyframe selection + text→frame retrieval
  brain/tools.py          vehicle tools + safety envelope
  brain/safety.py         workload estimation, alert arbitration, reply gating
  brain/local_brain.py    Qwen2.5-VL 4-bit on MLX, prompted tool calling
  brain/claude_brain.py   Claude vision + native tool use
  brain/router.py         edge/cloud routing with circuit breaker
  server/session.py       per-client real-time orchestration and backpressure
web/                      dashboard (vanilla JS, AudioWorklet mic capture)
docs/CONCEPTS.md          study guide: every ML concept, its math, and interview questions
docs/VISION.md            use cases, market, and roadmap toward production ADAS
eval/                     Eyeblink8 feature extraction, LOPO evaluation/training, parity checks
tests/                    unit tests (TTC vs ground truth, EAR geometry, safety envelope…)
scripts/e2e_check.py      end-to-end test over the WebSocket with synthetic video + speech
```

## Tests

`scripts/setup.sh` installs the test and dataset tools too (`requirements-dev.txt`).

```bash
.venv/bin/python -m pytest -q          # unit tests
.venv/bin/python scripts/e2e_check.py  # full pipeline (server must be running)
```

## Safety note

DriveMind is a research prototype, not a certified safety system. Its warnings are advisory. The LLM never actuates anything directly: every vehicle action passes through deterministic validation.

## Roadmap

See [docs/VISION.md](docs/VISION.md): dataset benchmarks (DMD, NTHU-DDD, DriveLM), lane detection, latency work (prefix caching, streaming TTS), a learned router, and porting perception to Jetson Orin / TensorRT.

## License

DriveMind's code is licensed under the [GNU AGPL-3.0](LICENSE) (chosen to match its Ultralytics
YOLO dependency). Copyright (C) 2026 Ruchir Bhatia.

Model weights are downloaded by `scripts/setup.sh`, not distributed with this repo, and keep their own licenses:

| Component | License |
|---|---|
| Ultralytics YOLO11 (road detection) | AGPL-3.0 |
| Qwen2.5-VL-3B-Instruct (on-device VLM, 4-bit MLX conversion) | Qwen Research License: **non-commercial research/evaluation use only**; commercial use requires a license from Alibaba Cloud |
| Whisper large-v3-turbo (speech recognition) | MIT |
| MediaPipe Face Landmarker (driver monitoring) | Apache-2.0 |
| Silero VAD (voice activity detection) | MIT |
| CLIP ViT-B/32 (visual memory) | MIT |
| MLX, mlx-vlm, mlx-whisper | MIT |

The evaluation datasets (Eyeblink8: GPL-3.0; UTA-RLDD: please cite Ghoddoosian et al., CVPRW 2019;
LISA Traffic Light Dataset: CC BY-NC-SA 4.0, please cite Jensen et al., IEEE T-ITS 2016 and Philipsen
et al., ITSC 2015) are not redistributed here, and neither are model weights trained on them.
