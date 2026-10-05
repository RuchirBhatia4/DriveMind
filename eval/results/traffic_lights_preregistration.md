# Traffic-light state: pre-registered test (2026-10-05)

Written and committed before any test data was downloaded. The classifier under test
(`classify_traffic_light` in `drivemind/perception/road.py`, this commit) and the scoring
(`eval/traffic_light_eval.py`, this commit) are frozen: nothing changes after seeing results,
and any later change gets a new held-out test.

## Why

In the voice-answer check, the on-device model said "The light is red." while both signals facing
the car were green: at 448 px wide they are a few pixels. The perception code already finds the
lights with YOLO and guesses their colour, but the guess was unusable, so it wasn't given to the
model. On 136 lights from the 24 local BDD100K clips (labelled by eye, `eval/labels/traffic_lights_dev.json`),
that colour-count rule gave a wrong colour for 36 of 85 lit lights and a colour to 32 of 47 side views,
pedestrian signals and other objects. New York's housings are painted yellow, so side views read
as "yellow". Green-painted housings read as "green". At 640 px a lit red lamp looks pale orange,
the same colour as sunlit paint.

The new classifier decides night from the whole frame (night: hue of the glow decides), rejects
square boxes (pedestrian signals, signs), trusts unambiguous colour (lit green is far more saturated
than paint or sky, and paint is never pure red), and otherwise uses position: the unlit lamps of a
signal facing us are dark discs and the lit one isn't. When unsure it gives no answer. On the dev
lights it was designed on: 0 of 85 wrong colours, 12 of 85 no answer, 2 of 47 false colours.
That is in-sample and optimistic, hence this test.

## Test sets

1. **LISA** (cropped LISA Traffic Light Dataset, Kaggle `chandanakuntala/cropped-lisa-traffic-light-dataset`,
   San Diego, labelled by the dataset's authors). Every crop in the download is used: neither
   classifier was trained on any of it. Labels: go / goLeft / goForward = green, stop / stopLeft = red,
   warning / warningLeft = yellow. Each crop is scaled to 0.5x (the app sees 640-px-wide frames);
   crops whose long side is under 12 px after scaling are skipped, since YOLO at 640 px rarely finds
   such lights. There are no frames, so night is judged from the crop's own median brightness.
   This set only has lit lights facing the camera: it tests colour, not side-view rejection.
2. **BDD100K test**: 20 clips drawn with `random.Random(0).sample` from the sorted list of every clip in
   the Kaggle mirror the demo clips came from (`robikscube/driving-video-with-object-tracking`), minus
   the 24 dev clips. One frame every 2 s, resized to 640x360, YOLO11n with the app's settings
   (confidence 0.35, 640 px, class "traffic light"). Every detection is labelled by eye from a
   full-resolution (1280x720) crop, on sheets that show no classifier output, before either
   classifier is run on them: red / yellow / green (lit lamp visible), none (side view, back, unlit),
   pedestrian (pedestrian signal), other (not a traffic light), unsure (excluded). The labels are
   committed before scoring. Labelling is done by the coding assistant (Claude), not by dataset
   annotators: the label file and the script that rebuilds the sheets are in the repo, so anyone
   can check them.

## Metrics (both classifiers, each set, overall and per colour)

- Lit lights: right colour, **wrong colour** (the dangerous error), no answer.
- Not lit (side views, pedestrian signals, other objects): **false colour** (any colour given).

## Predictions

- P1: On both sets the new classifier gives a wrong colour less often than the old one.
- P2: On the BDD100K test, the new classifier's wrong-colour rate on lit lights is at most 5%, and
  its false-colour rate on not-lit objects is at most 10%.
- P3: Yellow is untested (the dev lights had no amber), so it will be the weakest colour on LISA.
- The new classifier answers less often: expect no answer for 10-30% of lit lights.
