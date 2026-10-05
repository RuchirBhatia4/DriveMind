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

## Addendum (same day, after the LISA result, before labelling or scoring the BDD100K test)

**Pre-registered LISA result: P1 failed.** Under the protocol above (night judged from each crop),
the new classifier gave a wrong colour to 1,526 of 27,521 lit lights (5.5%), against 140 (0.5%)
for the old rule. Right colour: 62.1% vs 68.0%. No answer: 32.3% vs 31.5%. Almost all the errors
(1,353) were red lights called yellow.

**Why.** LISA's crops are named by sequence (`ds` = day, `ns` = night). At night its camera renders
a lit red lamp with a yellow-white core and only a red rim (1,311 of the 1,353 errors were night
crops), and the night rule counted hues, so amber won. Separately, judging night from the crop sent
most daytime crops through the night rule too (dark housings), while the app judges night from
the whole frame.

**Revision (frozen in this commit).** At night, when hue can't separate red from amber, the glow's
position in a box tall enough to hold the housing decides: top = red, middle = amber. The orange
night "yellow" rule is gone. The day rules are unchanged, and the dev lights score the same.
LISA was used to design this, so **LISA is in-sample for the revision**. There, with night taken
from the sequence name as the app would: wrong colour 119 / 27,521 (0.4%) vs 140 (0.5%) for the
old rule. Right colour 60.4% vs 68.0%. No answer 39.1% vs 31.5%. Under the original crop-judged
night it is 207 (0.8%).

**The BDD100K test is unchanged** and becomes the held-out test for the revision: 20 clips already
drawn (seed 0), 141 detections collected, no classifier run on them, and the sheets not yet looked
at. It will be scored for the old rule, the pre-registered version (commit e6ed69d) and this
revision. P1 and P2 apply to both versions.

## Addendum 2 (after the BDD100K test)

**Pre-registered BDD100K result.** The 20 clips held 141 detections in 16 clips: 82 lit lights,
55 not lit, 4 unsure (excluded). A red ball shown with a green or amber turn arrow was labelled red,
because for going straight "green" would be the dangerous answer.

| Classifier | Lit: right | Lit: wrong colour | Lit: no answer | Not lit: false colour |
|---|---|---|---|---|
| Old colour counts | 52 | 27 (33%) | 3 | 15 / 55 |
| Pre-registered version (e6ed69d) | 52 | 20 (24%) | 10 | 10 / 55 |
| Revision (a1d20af) | 52 | 20 (24%) | 10 | 5 / 55 |

P1 held (20 < 27 wrong colours). **P2 failed for both versions**: 24% wrong colours against a 5% target.
19 of the 20 errors were red lights called **green**, all at one intersection in one clip
(`0104f3cf-0dbe2960`, 19 frames 2 s apart). The camera rendered that sky saturated cyan, inside the
hue range of a lit green lamp. P3 held on LISA: yellow was the weakest colour (40% right, against
59-63% for red and green).

**Revision 2 (this commit).** A lit green lamp must be one compact blob inside the box: not touching
its edges, at least 2 px wide, and at least half filled. That rejects sky at the edges and slivers
of sky between visors. The cost is coverage: the 19 dull red lamps under that sky now get no answer
rather than a wrong one. In-sample on every set now:

- test: 48 right, 1 wrong colour, 33 no answer; 4 / 55 false colours
- dev: 72 right, 0 wrong colours, 13 no answer; 2 / 47 false colours
- LISA: 0.4% wrong colours

A second change, counting only glare-bright cores at night, was tried and dropped: it raised LISA's
night red-called-yellow errors from 40 to 268.

**New held-out test (test2).** 20 more clips, drawn with `random.Random(1).sample` from the mirror
minus the dev and test clips. The protocol is otherwise the same: one frame every 2 s, the app's
YOLO settings, labelled by eye before scoring, with red ball + arrow labelled red.

Predictions for revision 2 on test2:
- wrong colour on lit lights at most 5%
- false colour on not-lit objects at most 10%
- no answer for 30-45% of lit lights
