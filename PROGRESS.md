# Progress log

## 2026-10-26
**This includes everything from PRs #25, #26, #27, #28 and #29.** Merging this one is enough; the others will close automatically. (`main` is contained in it.)

**Done** (roadmap tasks 23d and 23h)
- 23d - "nothing here" photos: the downloader now also fetches ~120 Open Images photos that people checked contain no bottle, key, keychain or shoe (`python -m claudeknows.detection_data download --negatives 120`; credited in CREDITS.md). Your own `claudeknows_photos/empty/` folder (optional) is picked up too. Training mixes in a fresh random pick of them each round, at most 15% of the photos (`--neg-frac`, 0 turns it off). These photos are never counted as test "positives". The report now has a line "false alarms per background-only photo (score 0.5)" next to the scores. Fixed along the way: tilting/turning a photo with no boxes crashed the augmentation.
- 23h - same-room videos stay together: `scene_groups` merges videos with a similar background (average small thumbnail); the train/test split keeps each scene wholly on one side so two videos of one room can't flatter the scores. It prints which videos were merged; if everything looks like one scene it falls back to the by-video split and says so. The report shows how many different scenes the own-footage test set has.
- 84 tests pass (6 new).

**Next**
- 23j (zoomed-window training), 23k (averaged weights), 23l (better metres), then Phase 2. All still wait for your first real spatial-notebook run for tuning.

**For Alex to decide**
- Nothing blocking. The scene-matching sensitivity (0.08) is a guess; if the printout shows videos from different rooms merged (or the same room not merged), tell me.
- Authorship check: all commits by Claude or Alex (checked below in the PR).

## 2026-10-21 (evening, with Alex)
- Alex merged everything up to PR #21; `main` is up to date and all tests pass.
- **Alex's decision on task 23 (web page): option (a).** Done:
  - `web/spatial.html`: a short page that explains what the spatial model does and links to
    the two notebooks, plus what to film.
  - `notebooks/try_spatial.ipynb`: quick "try it" notebook (a few minutes). It loads a
    detector you already trained (from Drive `claudeknows_models/detector.pt`, or you upload
    the file) and lets you upload photos to see boxes, labels and distances. No training.
  - `notebooks/spatial_colab.ipynb` now saves the trained detector to Drive
    `claudeknows_models/` (a dated copy and `detector.pt`) as well as downloading it.
- The full in-browser version is task 23i, parked until we see real results.

**Next**
- Alex runs the spatial notebook (first real training of the detector). Then 23c/23d/23g/23h,
  and Phase 2.

## 2026-10-21
**This includes everything from PRs #12, #13, #14, #15, #16, #17, #18, #19 and #20.** Merging this
one is enough; the others will close automatically. (They merged into this branch with no
conflicts, and `main` is contained in it.)

**Done** (roadmap tasks 23e and 23f)
- Task 23e - gentler "any angle" training (`claudeknows/detector.py`): random tilts are now
  at most 8 degrees (was 20) because tilted photos make the boxes sloppy. In their place the
  training now sometimes (15%) turns a photo exactly 90 degrees either way, as phone photos
  often are, which keeps boxes exact, and rarely (10%) flips it upside down. Switch them with
  `--rot-degrees` and `--rot90-p`. A test checks a 90-degree turn gives exactly the right box.
  The "try 0 / 8 / 20 degrees and keep the best" comparison needs real training, so it is for
  the Colab run (use the new comparison below to judge it).
- Task 23f - honest score ranges: `bootstrap_ci` re-draws whole videos/photos 1000 times and
  gives a 95% range for the overall and per-object scores; `python -m claudeknows.detector
  report` now prints it next to each score, e.g. "(range 40-70%)". Few videos means a wide
  range. `compare_runs` takes two runs on the same test photos and says "helped", "hurt" or
  "no clear difference"; use it for 23b/23e and later the task 27 quality gate. Tests: ranges
  are wider with 3 videos than 30, and the three verdicts.
- Not done: task 23 still waits on your answer below; 23c, 23d, 23g, 23h are still open.

**Next**
- 23c / 23d (cleaner labels), 23g (test-time augmentation), 23h (scene-aware split), then Phase 2.

**For Alex to decide**
- Task 23 (web page): OK for it to be a simple "how to try it" page linking to the Colab
  notebook (my suggestion), or do you want an in-browser version (much more work)?
- Authorship check: see the PR description.

## 2026-10-19
**This run builds on the open PR #14** (which itself includes #12 and the research PR #13), so
merging this one brings in tasks 19-22 too. Nothing from those was on `main` yet.

**Done** (roadmap tasks 23a and 23b)
- Task 23a - better detector training (`claudeknows/detector.py`): the learning rate now
  eases in over the first 200 steps and then eases down; large bumps in learning are
  clipped; the GPU uses half-precision maths (faster, same results; CPU is unchanged).
  15% of the training photos/videos (chosen by video, never by frame) are held back; after
  every epoch the model is scored on them, the best epoch's weights are kept, and training
  stops after 4 epochs with no gain. The real test photos are still untouched until the
  final report. The best epoch and per-epoch scores are saved inside `detector.pt`.
- Task 23b - mixing objects into photos (`claudeknows/composite.py`, off by default;
  `--composite 0.3` turns it on): with 30% chance it pastes 1-3 objects cut from OTHER
  training photos into a photo, with soft edges, never overlapping something already
  there, and adds their boxes. Meant to teach "bottle next to shoes" since most internet
  photos show only one kind of object. Added a step 6b to `notebooks/spatial_colab.ipynb`
  that trains a second detector this way and prints the same report, so the `together/`
  rows can be compared with step 6. Keep it only if it helps there.
- 64 tests pass on CPU (new: `tests/test_composite.py` and more detector tests). Nothing
  was trained for real here (no GPU, pretrained weights blocked), so there are no
  accuracy numbers yet.
- Authorship check: every commit on `main` and on this branch is by Claude
  <noreply@anthropic.com> or alexg-sketch (merges by GitHub), so running the notebook
  is fine.
- Note: I started this run by re-doing tasks 19-20 before noticing PR #12 already had
  them; I threw that duplicate work away, so nothing is doubled.

**Next**
- Task 23 (web page for the spatial model) needs your choice below. After that, Phase 2.
- Open research PRs #15-#17 add more ideas (cleaner labels, honest scores, merging boxes
  from several views); they are not merged into this branch.

**For Alex to decide**
- Task 23: the detector is too big to run comfortably inside a web page. OK to make the
  page a simple "how to try it" page that links to the Colab notebook (my suggestion),
  or do you want me to attempt an in-browser version (much more work, slower)?
- Please merge this PR, then run `notebooks/spatial_colab.ipynb` in Colab with "Run all"
  (about 10 epochs on a GPU) and tell me the numbers in the step 6 and 6b reports.

## 2026-10-16
**This includes everything from PR #12 (tasks 19-20) and PR #13 (research notes).**

**Done** (roadmap tasks 21 and 22)
- Task 21 - `claudeknows/distance.py`: how far away is each found object. No training.
  - Two independent estimates per object: (1) Depth Anything V2 **Small** (the relative
    model gives the near -> far order, the metric-indoor Small model gives rough metres);
    (2) the research idea: the object's known real size (bottle 22 cm, shoe 28 cm, keys
    8 cm) plus the photo's EXIF focal length. The second one only works for photos with
    EXIF (phone photos yes; video frames and downloaded photos no, then the depth model is
    used alone).
  - The number shown is the mean of the two; if they differ by more than 1.5x the output
    says "the two estimates disagree". Keys are always flagged as rough.
  - `python -m claudeknows.distance check` scores both against your tape-measured
    `distance/` photos in plain words: typical error in metres and %, how often the
    near/far order is right, and the error after a scale fix fitted leave-one-out (never
    on the photo being scored; those photos stay test-only).
  - `draw_results` draws boxes with "1. keys ~1.2 m" labels (nearest = 1).
- Task 22 - `notebooks/spatial_colab.ipynb`: one "Run all" notebook: Open Images photos with
  boxes -> your Drive footage (optional) -> OWLv2 auto-boxes + contact sheets -> train
  detector -> report (per object / size / position / video) -> distance check -> upload
  a photo and see boxes + distances -> download `detector.pt`.
- 56 tests pass on CPU (new: `tests/test_distance.py`, a notebook test). Depth models and
  OWLv2 are blocked in the sandbox, so tests use fake depth maps; nothing here has run on
  real models or data yet.
- Authorship check on main and this branch (`git log`): all commits are by Claude
  <noreply@anthropic.com> or alexg-sketch (merges by GitHub), so running the notebook is fine.

**Next**
- Task 23 (web page for the spatial model), and the research tasks 23a (training recipe:
  warm-up, best-epoch pick) and 23b (paste objects into training photos). Phase 2 after that.

**For Alex to decide**
- Please run `notebooks/spatial_colab.ipynb` in Colab (GPU) once; it is the first time the
  detector, OWLv2 and the depth models run for real, so send me the scores and the distance
  report (or let Cowork Claude read the output). It works without your Drive footage, but
  distance can only be checked once `distance/` photos exist.
- Still open from the research PR: try RF-DETR Nano later if the first detector scores on
  your own videos are weak?

## 2026-10-14
**Done** (roadmap tasks 19 and 20)
- Task 19 - `claudeknows/autobox.py`: draws boxes on photos nobody has boxed yet
  (Open Images keys and all of your footage) with OWLv2 using text prompts. Only
  confident boxes are kept (score 0.3 or more, overlapping duplicates merged). In a
  single-object folder only that object's boxes are kept; `together/` keeps all
  three. Photos with nothing confident are marked "rejected" and left out of
  training. `contact_sheet` makes a grid of photos with boxes drawn so you can
  glance over them. OWLv2 downloads in Colab only; tests use a fake detector.
- Task 20 - `claudeknows/detector.py`: Faster R-CNN (MobileNetV3) with a new head
  for water_bottle / keys / shoes. Training augmentation moves the boxes with the
  photo: flips, rotation up to 20 degrees, zoom 0.5x-1.5x, shifts, perspective,
  colour. The report gives mAP@0.5 per object, per source (internet vs your
  footage), by box size in the photo (far/middle/near), by position (centre/edge)
  and per video/photo of yours. `python -m claudeknows.detector train|report`.
- 49 tests pass on CPU (new: `tests/test_autobox_detector.py`, including a tiny
  2-batch training run). Nothing was run against real models or data here:
  pretrained weights and OWLv2 are blocked in the sandbox.
- Authorship check: all commits on main and this branch are by Claude
  <noreply@anthropic.com> or alexg-sketch (merges by GitHub), so running
  notebooks is fine. (No notebook changes yet: task 22 will add the one for this.)

**Next**
- Task 21 (distance with Depth Anything V2 Small), then task 22 (the Colab
  notebook that runs download, auto-box, train, evaluate and distance).

**For Alex to decide**
- Nothing blocking. Rotated boxes come out a bit loose (the box around the
  rotated corners); I kept the 20-degree limit from your notes.

## 2026-10-12 (later, with Alex)
**Alex's decision:** the long-term goal is a robot that walks around, collects
data and retrains itself. Added **Phase 2: phone as the robot's eyes** to
`ROADMAP.md` (tasks 24-29, starting after task 23). Alex films a walk, and the
rest is automatic: frames, labelling (two labellers must agree), retraining,
and a quality check against a fixed, human-checked test set before any new
version replaces the old one. Phases 3 (wheeled robot) and 4 (training on a
schedule) are noted but not planned in detail yet.

## 2026-10-12 (evening, with Alex)
- Alex ran `notebooks/objects_colab.ipynb` in Colab. Result: **95.5% of the
  133 held-out test photos right** after 8 rounds (91.7% after round 1). About
  665 photos in total, which matches internet photos only (keys have far fewer
  than bottles and shoes).
- Added the trained model as `models/objects_mobilenet.pt` (6 MB). It holds
  only the weights, the 3 object names and the accuracy history; no file names
  or personal data.
- Spot check: internet photos of keys, shoes and bottles come out 99-100% sure.
  A cat photo came out "shoes, 62.5%": like the first model, it must always
  pick one of its objects, even when the photo is none of them.

## 2026-10-12
**Done** (roadmap tasks 16 and 18)
- Task 16 - `notebooks/objects_colab.ipynb`: one "Run all" notebook that downloads
  the internet photos, adds your Drive footage if the folders exist (otherwise
  internet only), shows sample photos per object, trains, prints accuracy per
  object and per source, shows the photos it got wrong, lets you upload a photo to
  try, and downloads the model (`objects.pt`).
- Task 18 - `claudeknows/detection_data.py` (whole photos with boxes):
  - `download`: whole Open Images photos (max side 640 px, not crops) with EVERY
    bottle / footwear box in them (tiny boxes under 3% of the photo are ignored),
    saved with a `boxes.json` per object folder, plus credits. Key photos have no
    boxes in Open Images, so they are marked "pending" (task 19 will box them).
  - `import`: your Drive footage, now including `together/` (several objects in
    one shot) and `distance/` (photos named like `keys_1.0m.jpg`, read into
    `distances.json`, **test only, never trained on**). Videos become frames at
    640 px. Missing folders are fine. Everything unboxed stays "pending" and is
    not used for training until task 19 boxes it.
  - Splits are by video/photo, never by frame. `DetectionDataset` hands out
    images + boxes in the format PyTorch detectors expect (a test runs a tiny
    detector on it). `python -m claudeknows.detection_data summary` shows counts.
  - Small supporting change: the frame/photo importers accept a bigger `max_side`.
- All tests pass on CPU (new: `tests/test_detection_data.py`, notebook test).
  Your photos/videos still cannot be committed (`.gitignore` + test).
- Authorship check on main and this branch: all commits are by Claude
  <noreply@anthropic.com> or alexg-sketch (merges by GitHub), so running the
  notebook is fine.
- Not checked against the real internet this time (only the fake-data tests ran).

**Next**
- Task 19: auto-boxing with OWLv2 in Colab for key photos and all of your footage
  (plus a contact sheet to glance over), then task 20 (detector + training).

**For Alex to decide**
- Nothing blocking. Optional: run `objects_colab.ipynb` to see how well the plain
  classifier does on your footage; the spatial work does not wait for it.

## 2026-10-10 (with Alex)
**Alex's decision:** the main priority is now **spatial recognition**: (1) find
where each object is in a photo and box it, (2) say how far away it is, and
(3) recognise objects from any angle or position. Wrote the plan into
`ROADMAP.md` ("Notes from Alex" and tasks 18-23). The classifier notebook
(task 16) is still next, because it tests the photo/video pipeline that the
spatial model reuses. Task 17 is replaced by task 23.

**For Alex**
- Extra things to film (details in `ROADMAP.md`): walk toward/away from each
  object, put it off-centre and partly hidden, film 2-3 objects together in one
  shot (`together/`), and take a few tape-measured distance photos
  (`distance/keys_1.0m.jpg` etc.).

## 2026-10-09
**Done** (roadmap tasks 14 and 15)
- Task 14 - getting the photos:
  - `claudeknows/openimages.py` downloads internet photos for water bottle,
    keys and shoes, crops them to the object's box (bottle boxes / footwear
    boxes; keys use the whole photo), skips the ones that no longer exist
    (404), caps at 300 per object and writes `CREDITS.md`.
  - `claudeknows/frames.py` turns videos into photos: ~2 per second, blurry
    ones skipped, ffmpeg when installed (handles iPhone .MOV/HEVC and phone
    rotation), otherwise OpenCV.
  - `claudeknows/objects.py` imports your folder
    (`claudeknows_photos/water_bottle|keys|shoes`; missing/empty folders are
    fine), combines it with the internet photos, and splits train/test **by
    video**, never by frame. If an object has only one video there is nothing
    to hold out, so all of it is used for training and the report says "no
    test photos" for it.
  - Your photos/videos can never be committed: `claudeknows_photos/`, `data/`
    and video/HEIC files are in `.gitignore`, and a test checks that.
- Task 15 - the model: `claudeknows/transfer.py` + `train_objects.py` use a
  small pretrained network (MobileNetV3-small) with a new last layer for the
  3 objects. Class names are saved inside the checkpoint, so
  `python -m claudeknows.predict` and `python -m claudeknows.evaluate` work with
  it unchanged. Training first trains only the new layer, then fine-tunes all.
  The report gives accuracy per object, per source (internet vs your footage),
  a confusion matrix and the photos it got wrong.
- 33 tests pass on CPU (tiny fake data and a generated mini video).
  The CIFAR-10 model and web page are untouched.
- Authorship check on main and this branch: all commits are by Claude
  <noreply@anthropic.com> or alexg-sketch (merges by GitHub), so running the
  Colab notebook is fine.

**Checked against the real internet:** ran the downloader for real (cap 6 per object, 3 minutes, mostly scanning the huge label files): got 6 photos for each object, credits written. Bottles and keys look right; shoe crops are the weakest (many are blurry close-ups of a foot), so shoes may need a stricter size filter after the first Colab run.

**Next**
- Task 16: Colab notebook for the new objects (download photos, mount Drive
  for your videos, train, show accuracy and mistakes, download the model).
- Task 17: web page for the new model.

**For Alex to decide**
- Nothing blocking. Please don't run anything yet: the notebook for this
  (task 16) comes next run. When it is merged, upload your videos to
  `claudeknows_photos/` in Google Drive (a few videos per object, different
  places and angles; at least 2 videos per object so one can be held out for
  testing).

## 2026-10-07 (with Alex)
**Done**
- Alex ran the Colab notebook: **87.8% accuracy** on the 10,000 CIFAR-10 test
  photos after 20 epochs (61% after epoch 1, 84% after 10, still rising slowly
  at the end).
- Task 12 - added the trained model as `models/cifar10_smallcnn.pt` (1.2 MB).
- Task 13 - web page `web/index.html`: choose a photo (or take one on a phone)
  and it shows the guess, how sure it is, all 10 scores, and the 32x32 image
  the model actually sees. The network runs in plain JavaScript inside the
  page (`web/classifier.js`, batch-norm folded in by
  `claudeknows/export_web.py`), so it needs no server and the photo never
  leaves the device. Tests check the JavaScript gives the same answers as
  PyTorch. A real cat photo comes out "cat, 99.4%".
- Fixed: PRs #2-#4 had been merged into a side branch, not `main`. PR #5 was
  retargeted to `main`, and all PRs now target `main`.

**Noticed**
- A photo of a rocket comes out "ship, 98.4% sure". The model has to pick one
  of its 10 things and can be very confident even when the answer is
  none of them.

**Next**
- Nothing left on the roadmap except the parked decision layer (tasks 7-8).

**Alex's decision:** next, teach it 3 new objects (water bottle, keys, shoes)
using photos from the internet. Checked Google's Open Images: plenty of water
bottles and shoes, but they're often tiny in the photo, so the plan crops to the
object. Keys are thin: only ~100-150 downloadable, often cluttered. Added tasks
14-17 to `ROADMAP.md`. Friday: tasks 14 and 15.

**For Alex to decide**
- Optionally add ~30 of your own photos of your keys to make keys more reliable.

## 2026-10-07
**Done**
- Task 11 - wrote `README.md`: install, train (incl. fake-data smoke run and
  the Colab notebook), evaluate, predict on one image, tests, and layout.
  The "customise actions" part is left out because the decision layer is parked.
- Checked the README commands against the real CLIs; all 12 tests pass on CPU.

**Next**
- All non-parked tasks are now done. Remaining: tasks 7-8 (decision layer, CLI), parked.
- The real CIFAR-10 accuracy is still unknown: it needs a Colab run of
  `notebooks/train_colab.ipynb`.

**Notes**
- PRs #2-#4 were merged into the branch `claude/2026-09-29`, not into `main`,
  because each one was stacked on the previous one. This PR now targets `main`
  directly and includes all of that work, so merging it brings `main` fully up
  to date. From now on every PR targets `main`, so they can be merged in any order.

**For Alex to decide**
- Run the Colab notebook and share the accuracy, or tell me to start the
  decision layer (tasks 7-8) or to try improving the model.

## 2026-10-05
**Done**
- Task 10 - `notebooks/train_colab.ipynb`: Colab notebook that clones the repo,
  installs deps, trains on CIFAR-10 (20 epochs), evaluates, lets you upload an
  image to classify, and downloads `checkpoint.pt`.
- Task 9 - tests already cover model, training step, evaluation and prediction
  (tasks 3-6); added `tests/test_notebook.py` checking the notebook is valid.
  All 12 tests pass on CPU.

**Next**
- Task 11 (README: install, train, predict). Tasks 7-8 stay parked.

**Notes**
- Builds on PR #3 (branch `claude/2026-10-02`), which builds on #2; merge those first.
- The notebook clones `main` by default, so it only works fully once the PRs
  are merged (or change `BRANCH` in the first code cell).

**For Alex to decide**
- Please run the notebook in Colab (GPU) and tell me the accuracy; that tells us
  whether the model needs to be bigger.

## 2026-10-02
**Done**
- Task 5 - `claudeknows/evaluate.py` (`python -m claudeknows.evaluate --checkpoint checkpoint.pt`):
  loads a checkpoint, prints overall accuracy, per-class accuracy and a confusion matrix.
- Task 6 - `claudeknows/predict.py` (`python -m claudeknows.predict IMAGE`):
  classifies one image file (any size, resized to 32x32) and prints the label,
  confidence and top-3 guesses. Also usable from Python via `predict_image`.
- `tests/test_evaluate_predict.py`: 3 new tests; all 11 tests pass on CPU.
  Smoke-ran train + evaluate with fake data.
- Decision layer (tasks 7-8) still parked per Alex's note.

**Next**
- Task 10 (Colab notebook for GPU training) and task 9 (fill any remaining test gaps), then README (11).

**Notes**
- Builds on PR #2 (branch `claude/2026-09-30`, itself on PR #1's work); merge #2 first.
- The real CIFAR-10 accuracy is still unknown: it needs the Colab run.

**For Alex to decide**
- Nothing blocking.

## 2026-09-30 (run 2)
**Done**
- Task 3 - `claudeknows/model.py`: `SmallCNN` (3 double-conv blocks with
  batch-norm, dropout, linear head; ~0.3M params) plus `count_parameters`.
- Task 4 - `claudeknows/train.py`: `python -m claudeknows.train` with
  `--epochs/--lr/--batch-size/--max-batches/--fake/--checkpoint`, GPU
  auto-detect, AdamW + cosine schedule, saves checkpoint (weights + history).
- `tests/test_model_train.py`: shape, param-range and a 2-batch smoke train.
  All 8 tests pass; a fake-data smoke run of the CLI works.
- Decision-layer tasks 7-8 remain parked per Alex's note.

**Next**
- Task 5 (evaluation: per-class accuracy, confusion matrix) and task 6
  (single-image prediction helper).

**Notes**
- Builds on PR #1 (branch `claude/2026-09-29`); merge that first.

## 2026-09-30
**Alex's decision:** for now the goal is just getting the model to understand
what it's looking at. I've recorded that under "Notes from Alex" in
`ROADMAP.md` and parked the decision-layer tasks (7 and 8). The next runs
focus on classification: CNN, training, evaluation, single-image prediction,
and the Colab notebook.

## 2026-09-29
**Done**
- Created `ROADMAP.md` (11 tasks) and this log.
- Task 1 – project setup: `claudeknows/` Python package, `requirements.txt`
  (torch, torchvision, pytest), `pytest.ini`, `.gitignore` (ignores `data/`,
  checkpoints, caches).
- Task 2 – dataset loader (`claudeknows/data.py`):
  - `get_dataloaders(...)` returns CIFAR-10 train/test loaders. Training images
    get random crop + horizontal flip; both splits are normalised with the
    standard CIFAR-10 mean/std.
  - `fake=True` swaps in torchvision's synthetic `FakeData` with the same
    shape (3x32x32, 10 classes), so tests and smoke runs need no download.
  - `CIFAR10_CLASSES` lists the 10 label names (airplane … truck).
  - 5 tests in `tests/test_data.py`, all passing on CPU.

**Notes**
- The CIFAR-10 download server is blocked from Claude's sandbox, so the real
  dataset has not been downloaded here. Tests use the fake mode. The real
  download will happen in the Colab notebook (task 10), where it works normally.

**Next**
- Task 3 (small CNN) and task 4 (training script with a tiny smoke run).

**For Alex to decide**
- What actions should the decision layer pick? Right now the plan is
  CIFAR-10's 10 labels (airplane, automobile, bird, cat, deer, dog, frog,
  horse, ship, truck). If you have a real use case in mind (e.g. "animal ->
  alert, vehicle -> log"), add it under "Notes from Alex" in `ROADMAP.md`.
  Otherwise I'll use a simple placeholder mapping.

## 2026-10-23
**Done** (tasks 23c and 23g; this PR also carries PR #25 and PR #26 - merging this one is enough)
- Task 23c - `python -m claudeknows.autobox fill data/detection`: for photos that already have boxes
  (including Open Images ones), look for the OTHER objects with OWLv2 and add a box (marked
  `owlv2-extra`) only if that object has no box yet, is >= 0.4 confident and does not overlap an
  existing box. Existing boxes are never changed; each photo is checked once. Reason: a bottle photo
  with shoes in the background used to teach the detector that those shoes are "nothing". The
  spatial Colab notebook now runs it right after auto-boxing; contact sheets show the added boxes.
- Task 23g - `detect_tta` in `detector.py` and `python -m claudeknows.detector report --tta flip|tiles`.
  `flip` = photo + mirrored + 1.3x bigger; `tiles` also runs four zoomed-in tiles to catch far-away
  (small) objects. Results are merged with `ensemble-boxes` (added to requirements.txt). The report
  prints the scores with and without it, with the usual confidence ranges. Off by default.
- 5 new tests (fill rules, tiles find a small object at the right place, mirror is exact); 80 pass.

**Next**
- 23d (background-only photos), 23h (scene-aware split), 23j (zoomed-window training).

**Notes**
- Tiles use "max" instead of "average" scores in the merge: otherwise an object seen by one tile
  only would be averaged down below the 0.5 cut-off.
- Whether to KEEP TTA/tiles needs a real trained detector: run the report with and without and
  look at the far-bin numbers.

**For Alex to decide**
- Nothing blocking. When you next run the spatial notebook, it will also run the new "fill" step.
  Authorship check: every commit on main and on this branch is by Claude or Alex (GitHub web-merges
  by "GitHub"), so running the notebook is fine.
