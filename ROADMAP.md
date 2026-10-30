# claudeknows roadmap

Image classifier + decision maker: look at an image, predict a label, then pick
an action from that label (and how confident the model is).

## Notes from Alex
<!-- Add notes here. Claude reads this section first on every run. -->
- 2026-09-30: Focus is only on getting the model to understand what it's looking at
  (accurate classification). The decision layer is parked until Alex says otherwise.
- 2026-10-07: Alex trained the model on Colab: 87.8% test accuracy. Keep the trained
  model in `models/cifar10_smallcnn.pt`. Every pull request must target `main`.
- 2026-10-07: Next goal: teach it 3 new objects: **water bottle, keys, shoes**.
  Alex wants to use photos from the internet, not his own (for now). Use Google's
  Open Images (openly licensed; the image host and label files are reachable from
  the Claude sandbox and from Colab):
  - Labels: water bottle `/m/0118n_9r` (~2,250 train photos), key `/m/03v3yw` (~300)
    plus keychain `/m/03lnq3` (~370), shoe `/m/06rrc` (~59,000). Label files:
    `https://storage.googleapis.com/openimages/v7/oidv7-train-annotations-human-imagelabels.csv`
    (2.7 GB, stream and grep; positives end in `,1.0`). Images:
    `https://open-images-dataset.s3.amazonaws.com/train/<ImageID>.jpg`.
  - Problem found: in most of these photos the object is tiny (a bottle on a meeting
    table, shoes on a person in a street). Crop to the object using the bounding-box
    files: "Bottle" boxes inside water-bottle photos, "Footwear" boxes for shoes.
    Keys have no boxes; use whole photos and expect keys to be the weakest object.
  - Only ~20-30% of key/keychain image IDs actually download (the rest return 404):
    expect ~100-150 key photos, many of them cluttered or keychain trinkets.
  - Alex's own photos can be added later in Google Drive (never commit them; the
    repo is PUBLIC).
  - 2026-10-07 (Alex, before going quiet until Friday): build everything assuming
    Alex WILL upload his own videos (and maybe photos) for ALL THREE objects, not
    just keys: `claudeknows_photos/water_bottle/`, `claudeknows_photos/keys/`,
    `claudeknows_photos/shoes/` in Google Drive. Training combines his
    videos/photos with the Open Images photos for each object, and must still work
    if a folder is missing or empty (internet photos only). Report results per
    source (his footage vs internet) so we can see what helps.
  - Alex will film a few short videos of each object (different places/angles). Turn videos into photos:
    ~2 frames per second, skip blurry frames (e.g. low variance of the Laplacian),
    handle phone formats incl. iPhone .MOV/HEVC (use ffmpeg when available, as in
    Colab; fall back to OpenCV `opencv-python-headless`; tests can generate a tiny
    video with OpenCV). Frames from one
    video are near-duplicates, so split train/test BY VIDEO, never by frame.
  - Start from a small pretrained network (transfer learning). Pretrained weights
    download in Colab but NOT in the Claude sandbox (download.pytorch.org is
    blocked), so tests must use `weights=None`.
  - The current CIFAR-10 model and web page must keep working.
- 2026-10-10: **Main priority from now on: spatial recognition.** Alex wants three things:
  1. **Where is it:** find each object in a photo and draw a box around it, with several
     objects in one photo (object detection), e.g. "keys bottom-left, bottle centre".
  2. **How far away:** say how far each found object is (nearer/farther, and rough metres).
  3. **Any angle or position:** recognise the objects however they're turned, near or
     far, centred or at the edge of the frame.
  Plan (keep it in this order; tasks 18-23):
  - Detection: fine-tune a small pretrained torchvision detector (e.g.
    `fasterrcnn_mobilenet_v3_large_fpn` or `ssdlite320_mobilenet_v3_large`, COCO weights)
    with a new head for water_bottle / keys / shoes. Train on WHOLE photos with boxes
    (not crops). Open Images boxes: "Bottle" boxes in water-bottle-labelled photos,
    "Footwear" boxes. Keys have no Open Images boxes.
  - Boxes for keys and for Alex's footage: auto-box in Colab with an open-vocabulary
    detector (OWLv2, `google/owlv2-base-patch16-ensemble`, Apache-2.0, via `transformers`)
    using text prompts ("keys", "water bottle", "shoe"); keep confident boxes only, and
    make a contact sheet so Alex can glance over them. Hugging Face is blocked in the
    Claude sandbox, so tests must mock it.
  - Distance: no training needed. Use a pretrained depth model in Colab: Depth Anything V2
    **Small** (Apache-2.0; the Base/Large versions are non-commercial, don't use them),
    `depth-anything/Depth-Anything-V2-Small-hf`, plus the metric-indoor Small variant for
    rough metres. For each box, report its depth and order objects near->far. Be honest
    in outputs: near/far order is reliable, metres are approximate.
  - Any angle: training augmentation that keeps boxes correct (random scale/crop, flips,
    rotation up to ~20 degrees, perspective, colour), plus a report that breaks accuracy
    down by object size in the photo (near/far), position (centre/edge) and Alex's held-out
    videos.
  - What Alex will film (all in Google Drive, never committed):
    - `claudeknows_photos/<object>/` videos as before, now ALSO: walking toward and away
      from the object, object off-centre and at the edges of the frame, partly hidden.
    - `claudeknows_photos/together/`: videos with 2-3 of the objects in the same shot
      (most important for detection).
    - `claudeknows_photos/distance/`: single photos with the object at a tape-measured
      distance, distance in the filename, e.g. `keys_1.0m.jpg`, `shoes_2.5m.jpg`. Used
      only to TEST distance estimates, never to train.
  - The classifier work (tasks 14-16) is reused: same downloader, video frames,
    by-video split. Task 17 (web page for the classifier) is replaced by task 23.

- 2026-10-12: **Long-term goal: a robot that walks around, collects data, and retrains
  itself.** Phases: (1) finish spatial recognition (tasks 18-23); (2) **phone as the
  robot's eyes** (tasks 24-29, below); (3) a wheeled robot kit running the model live;
  (4) training on a schedule without Colab clicks. Only phase 2 is planned in detail.
  Phase 2 design rules:
  - Alex films a "walk" around the house on his phone and drops it in Google Drive
    under `claudeknows_photos/walks/`. Everything after that is automatic (one
    "Run all" in Colab for now; phase 4 removes the click).
  - Self-labelling can teach the model its own mistakes. Guard against it:
    - a FIXED "gold" test set that Alex has checked; never trained on, never changed
      by the loop;
    - only confident labels where two independent labellers agree (current model
      + OWLv2) are used for training;
    - a new model version replaces the old one ONLY if it beats it on the gold set
      and no object gets worse by more than a small margin;
    - every version is kept, dated, in Drive `claudeknows_models/`, so rollback is
      always possible.
  - Uncertain frames are set aside for Alex: a short review batch (aim for 20 or
    fewer per walk) of yes/no questions, not drawing boxes.
  - The loop writes a plain-language summary (what it learned, gold-set scores before
    and after, promoted or not, review batch size) saved to Drive next to the models.
  - Never commit Alex's footage, frames, labels from his footage, or the gold set to
    the public repo.
  - Start phase 2 only after task 23 is done.
- 2026-10-21: Web page for the spatial model = **option (a)**: a simple page
  (`web/spatial.html`) linking to the Colab notebooks, plus a quick "try it" notebook
  (`notebooks/try_spatial.ipynb`) that loads a saved detector instead of retraining. The
  training notebook saves the detector to Drive `claudeknows_models/` automatically. The full
  in-browser version (option b, task 23i) is PARKED until Alex has run the spatial notebook and
  we have seen real results; it is also the first step toward running the model on a robot.

## Tasks
- [x] 1. Project setup: package layout, `requirements.txt`, pytest config, `.gitignore`
- [x] 2. Dataset loader: CIFAR-10 train/test `DataLoader`s with normalisation + augmentation, plus a synthetic "fake" mode for offline tests
- [x] 3. Small CNN in PyTorch (a few conv blocks, ~100k–1M params) with a shape test
- [x] 4. Training script (`python -m claudeknows.train`): epochs, lr, batch size, device auto-detect, `--max-batches` for smoke runs, saves a checkpoint
- [x] 5. Evaluation script: overall accuracy, per-class accuracy, confusion matrix
- [x] 6. Prediction helper: load checkpoint, run on a single image file, return label + confidence
- [ ] 7. *(parked, see notes)* Decision layer: map (label, confidence) to an action via a config (with a "not sure" fallback below a confidence threshold)
- [ ] 8. *(parked, see notes)* End-to-end CLI: `image -> label -> action`
- [x] 9. Tests for model, training step, and evaluation (all run on CPU in seconds)
- [x] 10. Colab notebook for full GPU training that clones the repo, trains, evaluates, and saves the checkpoint
- [x] 11. README: how to install, train, run predictions (customising actions waits on the parked decision layer)
- [x] 12. Trained model in the repo (`models/cifar10_smallcnn.pt`, 87.8% test accuracy from Alex's Colab run)
- [x] 13. Web page to try it: upload a photo, see the label, confidence and the 32x32 view (`web/index.html`)
- [x] 14. Open Images downloader: for each object, get the photo list, download photos (skip 404s), crop to the object's box where one exists, save small JPEGs in one folder per object (`water_bottle/`, `keys/`, `shoes/`), cap ~300 per object, write a credits file (CC BY). Must also accept a folder of Alex's own photos AND videos in the same layout (videos -> frames, see notes)
- [x] 15. Transfer-learning model + training: small pretrained network (e.g. MobileNetV3-small from torchvision) with a new final layer sized to the number of folders; save the class names inside the checkpoint; `predict` and `evaluate` work with it; tests run on CPU with `weights=None` and a tiny generated folder
- [x] 16. Colab notebook for the new objects: download the photos (task 14), show a few per object, train, show accuracy per object and the photos it got wrong, download the model
- [ ] 17. *(parked: replaced by task 23, the web page for the spatial model)* Web page for the classifier
- [x] 18. Detection data: from Open Images, keep whole photos with their boxes (bottle boxes in water-bottle photos, footwear boxes), several boxes per photo allowed; save as images + one JSON of boxes per object folder; credits file; plus `together/` and `distance/` folders from Alex's Drive (see notes). Tests with a tiny generated dataset
- [x] 19. Auto-boxing with OWLv2 (Colab only): boxes for key photos and for every frame of Alex's footage, confidence threshold, contact sheet of boxes for a quick human look; mocked in tests
- [x] 20. Detector model + training: pretrained torchvision detector with a 3-object head; box-aware augmentation for "any angle"; evaluation: mAP@0.5 per object and per source, plus breakdown by box size (near/far), position (centre/edge) and held-out videos; tests with `weights=None`
- [x] 21. Distance: Depth Anything V2 Small (relative) + metric-indoor Small (rough metres) per detected box; near->far ordering; check against Alex's `distance/` photos and report the error in plain words **[research 2026-10-16, see RESEARCH.md]** Also compute a size-prior estimate `focal_px * real_size_m / max(box_w, box_h)` (`focal_px` from EXIF FocalLengthIn35mmFilm / 36 * image width; real sizes bottle 0.22 m, shoe 0.28 m, keys 0.08 m low-confidence; skip if no EXIF); report both estimates and flag disagreement; report near/far order accuracy separately from metres; any scale correction for the metric model is fitted leave-one-out on the `distance/` photos (they stay test-only).
- [x] 22. Colab notebook for the spatial model: download + auto-box + train + evaluate + distance check; upload a photo and see boxes, labels, confidence and distances drawn on it; download the model
- [x] 23. Web page for the spatial model: **done as option (a)** (2026-10-21): `web/spatial.html` explains it and links to `notebooks/try_spatial.ipynb` (quick, loads the saved detector from Drive `claudeknows_models/detector.pt`) and `notebooks/spatial_colab.ipynb` (training, now saves to Drive)
- [x] 23a. **[research 2026-10-16, see RESEARCH.md]** Detector training recipe: split a validation set off the train set (by video/photo); per epoch compute val mAP@0.5; keep best-epoch weights, stop after 4 epochs without improvement; linear warm-up (~200 iters) + cosine LR decay; mixed precision (`torch.autocast` + `GradScaler`, CUDA only so CPU tests are unchanged); `clip_grad_norm_` 10. Test set stays untouched for the final report. Test on CPU with `weights=None`
- [x] 23b. **[research 2026-10-16, see RESEARCH.md]** Multi-object composite augmentation: with p~0.3 paste 1-3 box crops (train split only, from other photos/classes) into a training image, random scale, feathered edges, reject overlaps (IoU > 0.1); add their boxes+labels; `--composite` flag; report mAP with vs without, especially on `together/` held-out videos, and keep it only if it helps. No `ultralytics` (AGPL)
- [x] 23c. **[research 2026-10-17, see RESEARCH.md]** Fill in missing boxes: in `autobox.py` add `fill_missing`: run OWLv2 with all three prompts on EVERY folder (web and own, also human-boxed ones); append a box (`by="owlv2-extra"`) only for a class the photo does not already have, score >= 0.4, IoU < 0.3 with all existing boxes; never touch human boxes; print counts per folder; include in contact sheets. Tests with the fake detector
- [x] 23d. **[research 2026-10-17, see RESEARCH.md]** Background-only photos: ~100-150 Open Images photos human-verified to contain none of the targets (image-level rows ending `,0.0` for bottle, key, keychain, shoe, footwear), plus Alex's optional `claudeknows_photos/empty/`; `DetectionDataset` accepts empty-box samples; cap at ~15% of each epoch; report "false alarms per negative photo at score 0.5" next to mAP; never in the test positives. CPU test with an all-negative batch
- [x] 23e. **[research 2026-10-18, see RESEARCH.md]** Rotation augmentation fix: in `train_transform` lower `RandomAffine` degrees from 20 to 8 (loose rotated boxes hurt above ~3 degrees, Kalra et al. ICCV 2021) and add exact 90-degree turns (`RandomApply` p=0.15 of `RandomRotation((90,90)/(270,270), expand=True)`) and `RandomVerticalFlip(p=0.1)`; CLI flags `--rot-degrees`, `--rot90-p`; test a 90-degree turn gives the exact expected box; ablation degrees 0/8/20 with and without 90-degree turns, keep the best on validation
- [x] 23f. **[research 2026-10-18, see RESEARCH.md]** Confidence ranges: `bootstrap_ci` in `build_report` resamples whole test videos/photos (1000 times, seed 0, numpy only) and prints the 2.5-97.5% range next to mAP per object and overall; A/B decisions (23b, 23d, 23e) use the paired difference and say helped / no clear difference / hurt; reuse in the task 27 quality gate. Test: range is wider with 3 groups than with 30
- [x] 23g. **[research 2026-10-19, see RESEARCH.md]** Test-time augmentation: `detect_tta` in `detector.py` (original + mirror + 1.3x scale, boxes mapped back, merged with `ensemble-boxes` `weighted_boxes_fusion`, iou_thr 0.55, skip_box_thr 0.05); `--tta` flag in evaluation and notebook demo; test that un-mirroring a box is exact; keep it as an option only if it raises overall mAP (with 23f range) and hurts no object; off by default and off in the web page *(code done 2026-10-23: `detect_tta`, `report --tta flip|tiles`; whether to KEEP it waits on a real run)* **[research 2026-10-22, see RESEARCH.md]** Also add a `tiles` option to `detect_tta`: besides the full photo, run 4 overlapping tiles (2x2 grid, each tile 60% of the photo side, ~20% overlap, each resized up to the model size; with `IMAGE_SIZE` 640x960 a tile is upscaled ~1.7x), map boxes back to the full photo, and merge everything with the same WBF call. This targets the far (small) bin where run 3 found only 28%. Report far / middle / near recall and mAP separately with and without tiles (with the 23f range); keep it only if far improves clearly and near does not get worse. CPU test: a fake detector that only sees objects covering >10% of its input finds a small box only through the tiles, and the box lands at the right place in the full photo.
- [x] 23h. **[research 2026-10-20, see RESEARCH.md]** Scene-aware split: in `detection_data.py` add `scene_groups(samples, threshold)` that merges Alex's videos showing the same background (descriptor = mean 16x16 RGB thumbnail or colour histogram per video, numpy/PIL only; single-linkage / union-find) and make `split_detection` keep each merged scene wholly on one side; Open Images photos stay one group each; print videos -> scenes; if all merge into one scene, fall back to the by-video split and say so. Reports show the number of test scenes next to own-footage scores. Test: two fake videos with the same background always land on the same side.
- [x] 23j. **[research 2026-10-22, see RESEARCH.md]** Zoomed-window training (pairs with the `tiles` option of 23g): in `train_transform`'s `apply`, with p=0.25 (flag `--zoom-crop-p`, default 0 until it proves itself) crop a random window of 50-70% of the photo side (same aspect ratio) that fully contains at least one box (try up to 10 draws; otherwise skip), keep boxes that stay >=60% inside, clip them, and let the model's resize scale the window up. Objects then look in training like they do inside a tile at test time. Needs no new library (`torch` slicing + existing `ClampBoundingBoxes`/`SanitizeBoundingBoxes`). Test: a box fully inside the window keeps the exact shifted coordinates. Ablation with `--zoom-crop-p` 0 / 0.25 on the validation set: keep the setting that raises far-bin mAP without lowering overall mAP (paired difference from 23f). Run it together with `--tta tiles` for a fair comparison.
- [x] 23k. **[research 2026-10-24, see RESEARCH.md]** Weight moving average (EMA) option: in `train_detector` add `ema_decay=0.0` (flag `--ema-decay`, 0 = off) using `torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(decay), use_buffers=True)`; call `ema.update_parameters(model)` after `sched.step()` (copy raw weights until the warm-up ends, then average); validate on `ema.module`, keep best-epoch weights from the EMA copy, save EMA weights in the checkpoint. Try 0.995 (0.999 if runs are long). Test on CPU with `weights=None`: EMA weights differ from raw and lie between start and current values; decay 0 changes nothing. Ablation `--ema-decay` 0 / 0.995 with the 23f range: keep it on only if overall mAP does not drop and the validation curve is steadier.
- [ ] 23l. **[research 2026-10-25, see RESEARCH.md]** Better metres from two guesses (builds on done task 21): in `distance.py` `check_distances` also score the geometric mean `sqrt(depth*size)` and a log-space blend `exp(w*log(size)+(1-w)*log(depth))` with `w` fitted leave-one-out on `distance/` (grid 0..1 step 0.1); show the winner via a constant `BLEND_W` (default 0.5 until 10+ photos exist); when a photo has 2+ detections with a size prior (not keys), fit one per-photo scale `median(size_m/depth_m)`, apply it to all that photo's depth-model metres and flag `calibrated_by_photo`. Near/far order logic unchanged. Tests on fake maps: geometric mean of 1 m and 4 m is 2 m; the blend picks w=1 when depth is wildly off; a keys box in a photo with a bottle lands right when the metric map is uniformly 2x too large.
- [x] 23n. **[research 2026-10-29, see RESEARCH.md]** Resumable detector training: `train_detector(..., resume_path=None)` and `--resume PATH`; after every epoch write (temp file, then `os.replace`) model, optimiser, scheduler, GradScaler, EMA, epoch, step, histories, best epoch and best weights, negatives RNG state and the run settings; if the file exists and the settings match, continue from the next epoch, otherwise say so and start fresh; remove it when training ends normally. The spatial notebook passes a per-run file in Drive `claudeknows_models/` when Drive is on. CPU test with `weights=None`: 2+2 epochs with a resume in between gives the same final weights (1e-5) as 4 straight; mismatching settings are ignored with a message; no flag = unchanged behaviour.
- [x] 23o. **[research 2026-10-29, see RESEARCH.md]** Parallel photo loading: `--workers` (default 2, tests use 0) for the detector training loader (and any loader in `predict_samples`/`mean_ap`), `pin_memory` on CUDA, and a `worker_init_fn` that reseeds every random generator the transforms use (composite, zoom, rotation) from `seed + worker_id` so workers do not repeat the same draws. CPU test with 2 workers: draws differ between workers and the loop finishes; `--workers 0` reproduces today's results. The notebook prints the time of the first epoch.
- [ ] 23i. *(parked: Alex decides after the first real spatial-notebook results)* Full in-browser version of the spatial model (option b). Plan from research 2026-10-18: `export_web.py` gets `export_detector_onnx` (`torch.onnx.export(model, ([x],), path, opset_version=17, dynamo=False)`, fixed 3x480x640 input; verified to export and load in onnxruntime, ~76 MB float32); test on a tiny `weights=None` model; parity check on the trained checkpoint (same boxes within 1 px for scores > 0.5). Browser: `onnxruntime-web` (MIT) for the detector, Transformers.js + `onnx-community/depth-anything-v2-small` (Apache-2.0) for depth, size-prior distance in plain JS. Model file as a GitHub Release asset, not in git. If parity fails or it is too slow, keep the link-to-notebook fallback. No Ultralytics/YOLO (AGPL).
- [ ] 23m. **[research 2026-10-27, see RESEARCH.md]** Per-object score cut-offs: in `autobox.py` make `THRESHOLD`/`FILL_THRESHOLD` dicts keyed by prompt label (default = today's numbers for every object, `--threshold` still sets all; lower keys only if the contact sheet shows keys boxes being lost); in `detector.py` add `best_thresholds(model, val_samples)` (F1 sweep 0.05-0.95, IoU 0.5 matching, validation photos only, an object with fewer than 20 val boxes keeps 0.5), save as `thresholds` in the checkpoint, and let `detect`/`detect_tta` take `min_score=None` = use the saved cut-off else 0.5; report prints precision, recall and cut-off per object with the 23f range. Old checkpoints must still load. Test with a fake model: a low-scoring true object gets a low cut-off, a high-scoring false one a high cut-off.
- [ ] 24. *(Phase 2)* Walk import: read new videos from `claudeknows_photos/walks/`, turn them into frames (reuse frames.py), drop near-duplicate frames (perceptual hash), and keep a manifest so each walk is processed once
- [ ] 25. *(Phase 2)* Gold test set: build it from Alex's checked footage and `distance/` photos plus a few held-out walks; freeze it with file hashes; evaluation always reports scores on it; a check refuses to train on any gold image
- [ ] 26. *(Phase 2)* Auto-labelling of walk frames: current detector + OWLv2 must agree (same object, overlapping boxes, both confident); agreed frames become training labels; disagreements and low-confidence frames go into a small review batch for Alex (yes/no questions, aim for 20 or fewer per walk); cap the share of self-labelled data in training **[research 2026-10-21, see RESEARCH.md]** Add a consistency rule after the agreement test: keep a box only if a same-label box with IoU > 0.3 exists in at least 2 of the 4 nearest sampled frames of the same walk (numpy only); failed boxes go to the review batch only if confident, else dropped; log how many boxes each rule removed. Test: fake 5-frame walk, one steady box kept, one single-frame box dropped.
- [ ] 27. *(Phase 2)* Retrain + quality gate: train a candidate on all approved data, compare to the current model on the gold set, promote only if better overall and no object worse by more than ~2 points; save every version dated in Drive `claudeknows_models/`; one-line rollback
- [ ] 28. *(Phase 2)* Learning-loop notebook: one "Run all" that does 24-27, shows the review batch with yes/no buttons, and writes a plain-language summary (scores before and after, promoted or not, what to review) to Drive
- [ ] 29. *(Phase 2)* Drift report: track gold-set scores, label counts per object and how often self-labels were later corrected, across loop runs; warn in the summary if anything trends the wrong way
