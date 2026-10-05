"""Traffic-light state: the old colour-count rule vs the lamp-position classifier in road.py.

Three sets:
  dev   136 lights YOLO found in the 24 local BDD100K clips (one frame every 2 s), labelled
        by eye from full-resolution crops. The classifier was designed on these, so its
        dev score is optimistic.
  lisa  The cropped LISA Traffic Light Dataset (San Diego), labelled by its authors. Never
        looked at during design. It has amber lights, which dev has none of.
  test  Lights in 20 BDD100K clips drawn at random after the classifier was frozen, labelled
        by eye before either classifier ran on them.
Protocol, fixed before any test data was downloaded: eval/results/traffic_lights_preregistration.md

    .venv/bin/python -m eval.traffic_light_eval collect   # test clips -> detections + labelling sheets
    .venv/bin/python -m eval.traffic_light_eval dev|lisa|test
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from drivemind.config import ROOT, settings
from drivemind.perception.road import classify_traffic_light, is_night
from eval.bdd_clips import DATASET, PREFIX, get_clips
from eval.fcw_footage_eval import CLIPS as DEV_CLIPS

LABELS = ROOT / "eval" / "labels"
RESULTS = ROOT / "eval" / "results"
SHEETS = ROOT / "data" / "traffic_light_sheets"
LISA = ROOT / "data" / "lisa_crops"
LISA_DATASET = "chandanakuntala/cropped-lisa-traffic-light-dataset"
LISA_MAP = {"go": "green", "goLeft": "green", "goForward": "green", "stop": "red", "stopLeft": "red",
            "warning": "yellow", "warningLeft": "yellow"}
LIT = ("red", "yellow", "green")
UNLIT = ("none", "pedestrian", "other")  # side view or unlit, pedestrian signal, not a traffic light
STEP_S, TEST_SEED, TEST_N = 2.0, 0, 20
LISA_SCALE, LISA_MIN_H = 0.5, 12  # app frames are 640 px wide; YOLO rarely finds lights under ~12 px


def old_classify(crop_bgr: np.ndarray) -> str | None:
    """The previous road.py rule, kept verbatim for comparison: count bright, saturated
    pixels in red / yellow / green hue bands."""
    if crop_bgr.size == 0:
        return None
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    bright = (hsv[..., 1] > 90) & (hsv[..., 2] > 140)
    h = hsv[..., 0]
    counts = {
        "red": int(np.count_nonzero(bright & ((h < 10) | (h > 165)))),
        "yellow": int(np.count_nonzero(bright & (h >= 15) & (h <= 35))),
        "green": int(np.count_nonzero(bright & (h >= 45) & (h <= 95))),
    }
    color, n = max(counts.items(), key=lambda kv: kv[1])
    return color if n >= max(6, 0.02 * crop_bgr.shape[0] * crop_bgr.shape[1]) else None


def frame640(cap: cv2.VideoCapture, t: float) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * 30)))
    ok, big = cap.read()
    if not ok:
        raise RuntimeError(f"no frame at {t}s")
    return cv2.resize(big, (640, 360), interpolation=cv2.INTER_AREA)


def score(items: list[tuple[str, str | None, str | None]]) -> dict:
    """items: (truth, old output, new output). Lit lights: right colour / wrong colour / no answer.
    Not lit (side views, pedestrian signals, other objects): any colour is a false colour."""
    out = {}
    for name, k in (("old", 1), ("new", 2)):
        lit = [(it[0], it[k]) for it in items if it[0] in LIT]
        unlit = [(it[0], it[k]) for it in items if it[0] in UNLIT]
        per = {c: {"n": sum(t == c for t, _ in lit), "right": sum(t == c and g == c for t, g in lit),
                   "wrong": sum(t == c and g is not None and g != c for t, g in lit)} for c in LIT}
        out[name] = {
            "lit": len(lit), "right": sum(g == t for t, g in lit),
            "wrong_colour": sum(g is not None and g != t for t, g in lit),
            "no_answer": sum(g is None for _, g in lit),
            "unlit": len(unlit), "false_colour": sum(g is not None for _, g in unlit),
            "per_colour": per,
        }
    return out


def table(title: str, s: dict, note: str = "") -> str:
    def pct(a, b):
        return f"{a}/{b} ({100 * a / b:.1f}%)" if b else "0/0"
    rows = [f"### {title}", "", note, "",
            "| Classifier | Lit: right colour | Lit: **wrong colour** | Lit: no answer | Not lit: **false colour** |",
            "|---|---|---|---|---|"]
    for name, label in (("old", "Old (colour counts)"), ("new", "New (lamp position)")):
        r = s[name]
        rows.append(f"| {label} | {pct(r['right'], r['lit'])} | {pct(r['wrong_colour'], r['lit'])} | "
                    f"{pct(r['no_answer'], r['lit'])} | {pct(r['false_colour'], r['unlit'])} |")
    rows += ["", "Per colour (right / wrong colour of n): " + "; ".join(
        f"{name} {c} {s[name]['per_colour'][c]['right']}/{s[name]['per_colour'][c]['wrong']} of {s[name]['per_colour'][c]['n']}"
        for name in ("old", "new") for c in LIT), ""]
    return "\n".join(r for r in rows if r is not None)


def run_bdd(name: str) -> dict:
    data = json.loads((LABELS / f"traffic_lights_{name}.json").read_text())
    if any(d["label"] is None for d in data):
        raise SystemExit(f"{name}: some lights are not labelled yet")
    clips = sorted({d["clip"] for d in data})
    paths = dict(zip(clips, get_clips(clips)))
    caps, frames, items = {}, {}, []
    for d in data:
        if d["label"] == "unsure":
            continue
        key = (d["clip"], d["t"])
        if key not in frames:
            cap = caps.setdefault(d["clip"], cv2.VideoCapture(paths[d["clip"]]))
            frames[key] = frame640(cap, d["t"])
        f = frames[key]
        x1, y1, x2, y2 = d["box"]
        crop = f[int(y1):int(y2), int(x1):int(x2)]
        items.append((d["label"], old_classify(crop), classify_traffic_light(crop, is_night(f))))
    s = score(items)
    s["excluded_unsure"] = sum(d["label"] == "unsure" for d in data)
    s["clips"] = len(clips)
    return s


def run_lisa() -> dict:
    if not glob.glob(str(LISA / "**" / "*.jpg"), recursive=True):
        LISA.mkdir(parents=True, exist_ok=True)
        subprocess.run([str(ROOT / ".venv" / "bin" / "kaggle"), "datasets", "download", LISA_DATASET,
                        "-p", str(LISA), "--unzip", "-q"], check=True)
    items, skipped = [], 0
    for path in sorted(glob.glob(str(LISA / "**" / "*.jpg"), recursive=True)):
        truth = LISA_MAP.get(Path(path).parent.name)
        img = cv2.imread(path)
        if truth is None or img is None:
            continue
        h, w = img.shape[:2]
        small = cv2.resize(img, (max(1, round(w * LISA_SCALE)), max(1, round(h * LISA_SCALE))), interpolation=cv2.INTER_AREA)
        if max(small.shape[:2]) < LISA_MIN_H:
            skipped += 1
            continue
        items.append((truth, old_classify(small), classify_traffic_light(small)))  # no frame, so night is judged from the crop
    s = score(items)
    s["skipped_too_small"] = skipped
    return s


def list_mirror_clips() -> list[str]:
    cache = ROOT / "data" / "bdd100k" / "mirror_clips.txt"
    if cache.exists():
        return cache.read_text().split()
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    names, token = [], None
    while True:
        r = api.dataset_list_files(DATASET, page_token=token, page_size=200)
        names += [f.name for f in r.files if f.name.startswith(PREFIX) and f.name.endswith(".mov")]
        token = r.next_page_token
        if not token:
            break
        time.sleep(2)  # the listing API rate-limits bursts
    clips = sorted(Path(n).stem for n in names)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text("\n".join(clips) + "\n")
    return clips


def collect() -> None:
    """Draw the test clips, find lights with the app's YOLO settings, and write labelling sheets
    cropped from the full 1280x720 frame. The sheets show no classifier output."""
    from ultralytics import YOLO

    pool = sorted(set(list_mirror_clips()) - set(DEV_CLIPS))
    clips = sorted(random.Random(TEST_SEED).sample(pool, TEST_N))
    print(f"{len(pool)} clips in the mirror outside dev; drew {len(clips)}")
    paths = dict(zip(clips, get_clips(clips)))
    model = YOLO(settings.yolo_model)
    data = []
    for clip in clips:
        cap = cv2.VideoCapture(paths[clip])
        dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / 30
        t = 0.0
        while t < dur - 0.5:
            f = frame640(cap, t)
            r = model(f, conf=settings.yolo_conf, imgsz=settings.yolo_imgsz, classes=[9], verbose=False)[0]
            for x1, y1, x2, y2 in r.boxes.xyxy.tolist():
                data.append({"clip": clip, "t": t, "box": [round(v, 1) for v in (x1, y1, x2, y2)], "label": None})
            t += STEP_S
    LABELS.mkdir(parents=True, exist_ok=True)
    out = LABELS / "traffic_lights_test.json"
    if out.exists() and any(d["label"] for d in json.loads(out.read_text())):
        raise SystemExit(f"{out} already has labels; not overwriting")
    out.write_text(json.dumps(data, indent=1))
    sheets(data, "test", {c: p for c, p in paths.items()})
    print(f"{len(data)} lights in {len({d['clip'] for d in data})} of {len(clips)} clips -> {out}")


def sheets(data: list[dict], tag: str, paths: dict[str, str], per: int = 40, cols: int = 10) -> None:
    SHEETS.mkdir(parents=True, exist_ok=True)
    tw, th, caps = 110, 190, {}
    tiles = []
    for i, d in enumerate(data):
        cap = caps.setdefault(d["clip"], cv2.VideoCapture(paths[d["clip"]]))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(d["t"] * 30)))
        ok, big = cap.read()
        x1, y1, x2, y2 = [v * 2 for v in d["box"]]
        pw, ph = 0.3 * (x2 - x1), 0.15 * (y2 - y1)
        c = big[max(0, int(y1 - ph)):int(y2 + ph), max(0, int(x1 - pw)):int(x2 + pw)]
        k = min((tw - 4) / c.shape[1], (th - 24) / c.shape[0])
        c = cv2.resize(c, (max(1, int(c.shape[1] * k)), max(1, int(c.shape[0] * k))), interpolation=cv2.INTER_NEAREST)
        t = np.full((th, tw, 3), 40, np.uint8)
        t[22:22 + c.shape[0], 2:2 + c.shape[1]] = c
        cv2.putText(t, str(i), (3, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        tiles.append(t)
    for k in range(0, len(tiles), per):
        page = tiles[k:k + per] + [np.zeros((th, tw, 3), np.uint8)] * (-len(tiles[k:k + per]) % cols)
        grid = np.vstack([np.hstack(page[j:j + cols]) for j in range(0, len(page), cols)])
        cv2.imwrite(str(SHEETS / f"{tag}_{k // per}.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 92])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("set", choices=["collect", "dev", "lisa", "test"])
    which = ap.parse_args().set
    if which == "collect":
        return collect()
    s = run_lisa() if which == "lisa" else run_bdd(which)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"traffic_lights_{which}.json").write_text(json.dumps(s, indent=1))
    notes = {
        "dev": "Designed on these lights, so this is an optimistic, in-sample score.",
        "lisa": f"Every crop scaled to {LISA_SCALE:g}x (app frames are 640 px wide); crops under {LISA_MIN_H} px "
                f"skipped ({s.get('skipped_too_small', 0)}). No side views in this set: it only has lit, facing lights.",
        "test": f"{s.get('clips', '?')} clips drawn after the classifier was frozen; {s.get('excluded_unsure', 0)} "
                "lights labelled unsure and excluded.",
    }
    print(table(which, s, notes[which]))


if __name__ == "__main__":
    os.environ.setdefault("YOLO_VERBOSE", "False")
    main()
