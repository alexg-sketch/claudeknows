# Research log

Daily notes from the research advisor. The dev routine reads the roadmap tasks tagged
`[research YYYY-MM-DD]`; this file holds the reasoning and sources behind them.

## 2026-10-29

**Summary.** Started from PR #32 (newest open branch; it includes #25 to #31, so merging it is enough). Searched for video-box propagation
(SAM 2), small-data Faster R-CNN fine-tuning results and free-Colab limits. No new detector, labeller or depth change is worth making.
Reading the training code against Colab's limits turned up two practical gaps instead. (1) Training cannot pick up where it stopped: if the free
Colab session drops (idle timeout, 12-hour cap, quota), the whole run is lost, and the roadmap now plans many 15-epoch ablation runs
(EMA, zoom windows, rotation, composites). (2) The detector's data loader runs in one process (no `num_workers`), so the GPU waits while the
CPU decodes JPEGs and applies rotations, pastes and zooms. Both are plumbing, not new science, and cost few lines.

### Recommendations

**1. Resumable training (new task 23n; small effort, low risk).**
- What: save the full training state to a file after every epoch and, if that file exists, continue from it. Free Colab gives a T4, roughly a
  12-hour hard cap and an idle timeout (third-party figures; Google does not guarantee free GPUs), so a dropped session is normal, and every
  ablation planned in 23e / 23j / 23k / 23b is a fresh 15-epoch run.
- How: `train_detector(..., resume_path=None)` (flag `--resume PATH`, default none = today's behaviour). After each epoch `torch.save` (to a temp file, then
  `os.replace`, so a cut-off write cannot corrupt it) a dict with: model state, optimiser, scheduler, GradScaler, EMA state (if on), `epoch`, `step`,
  `history`, `val_history`, `best`, `best_epoch`, `best_state`, the Python `random.Random` state used for negatives, and the settings that must match
  (epochs, batch size, lr, zoom/rot/composite/ema values). If the file exists and the settings match, load all of it and start the loop at `epoch+1`; if
  they differ, say so and start fresh. Delete or ignore the file when training ends normally. The notebook passes `--resume
  /content/drive/MyDrive/claudeknows_models/resume_<run-name>.pt` when Drive is on, so "Run all" after a disconnect simply continues.
- Acceptance: CPU test with `weights=None`: train 4 epochs in one go versus 2 epochs, stop, resume for 2 more; the final weights match to 1e-5
  (use `max_batches`, fixed seed, workers 0); a resume file with different settings is ignored with a message; no `resume_path` changes nothing. The data
  shuffle after resume may differ from an unbroken run in the epoch it restarts; say so in the docstring rather than chase exact equality.
- Expected gain: no change to accuracy; saves up to a whole run of Colab quota after each disconnect. Effort: about 40 lines. Risk: low; resume files
  are large (model + optimiser + best copy, a few hundred MB), so keep one per run and overwrite it. Licence: none (PyTorch BSD-3).

**2. Load photos in parallel (new task 23o; tiny effort, low risk).**
- What: `train_detector`'s `epoch_loader` builds `DataLoader(ds, batch_size, shuffle=True, collate_fn=collate)` with the default `num_workers=0`, so JPEG
  decoding, affine rotation, composite pasting and zoom windows all happen in the training process between GPU steps. Colab VMs usually have 2
  CPU cores (this sandbox shows 4), so 2 worker processes is the standard fix. The classifier loaders in `data.py` and `evaluate.py` already use 2.
- How: add `num_workers=2` (flag `--workers`, default 2; 0 for tests) to `train_detector`, `predict_samples` / `mean_ap` loaders if they use a loader, plus
  `pin_memory=True` when CUDA and `persistent_workers=False` (the loader is rebuilt every epoch because negatives are re-sampled, so persistent workers
  would not help). Seed each worker: pass a `worker_init_fn` (and `generator`) that derives from `seed + worker_id`, because `CompositePaste` and the
  random zoom/rotation use their own RNGs and workers would otherwise repeat the same "random" draws every epoch. If the transform holds one `random.Random`,
  copy-on-fork gives every worker the same stream: reseed it in `worker_init_fn`.
- Acceptance: CPU test with `weights=None` and `num_workers=2`: two workers do not return identical augmentation draws for identical inputs; the loop still
  finishes; `--workers 0` reproduces today's results with the same seed. Time one epoch with 0 vs 2 workers on Colab and print it in the notebook so the gain is visible.
- Expected gain: shorter epochs, most of all with composite and zoom turned on; no change to accuracy. Effort: about 10 lines. Risk: the RNG-per-worker
  point above is the one thing that can quietly make augmentation less random. Licence: none.

### Considered but not recommended
- **SAM 2 box propagation to label video frames (Apache-2.0 code, tools such as SAM2YOLO and Roboflow's video tracker):** a real way to label one frame and
  follow the object through the clip, but it needs a hand-drawn seed box per clip and a large model next to the detector; OWLv2 plus the consistency rule
  in task 26 already covers this more cheaply. Found no 2026 study of its quality on small objects. Revisit if OWLv2 boxes on Alex's footage look poor on the contact sheets.
- **More backbone-unfreezing / resolution experiments for torchvision MobileNetV3 Faster R-CNN:** the search returned only API docs and no results for
  this size; nothing to act on.
- **Mixed precision / batch-size tweaks:** already done in 23a; a T4 gains nothing from bf16.

### For Alex to decide
- Nothing new. Still open: RF-DETR / DEIMv2 (10-16) and Depth Anything V3 Small vs V2 Small (10-19).

### Sources
- SAM 2 for video annotation (tool overview): https://www.labellerr.com/blog/sam-2-video-annotation/
- SAM2YOLO, labels from video with SAM 2: https://app.readytensor.ai/publications/a-software-tool-to-generate-labeled-image-data-from-input-videos-using-sam2-HtrHlVmlQ7La
- Free Colab limits (third-party summary, unverified against Google): https://www.hivenet.com/post/google-colaboratory-gpu-complete-guide-to-free-cloud-gpu-access-and-limitations
- torchvision Faster R-CNN MobileNetV3 docs: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn.html
- PyTorch data loading (`num_workers`, `worker_init_fn`): https://pytorch.org/docs/stable/data.html (could not be fetched from the sandbox today; standard documented behaviour)

## 2026-10-27

**Summary.** Started from PR #30 (newest open branch; it includes #25 to #29, so merging it is enough). Searched for noisy auto-box handling,
threshold choice for detectors, and phone-camera depth calibration. No new detector, depth or labeller change is worth making. One small
gap turned up in the code itself: every object shares one cut-off, both when OWLv2 draws the training boxes (0.3, and 0.4 for the "fill in"
step) and when the trained detector decides what to show (0.5). A 2026 study of open-vocabulary detectors found the best cut-off differed a
lot between models (0.10 to 0.41), because these scores are not comparable between objects. Keys are the hardest, least-seen object, so they
probably score lower than bottles and get dropped by a cut-off tuned for bottles.

### Recommendations

**1. Per-object score cut-offs (new task 23m; small effort, low risk).**
- What: pick one cut-off per object (water_bottle, keys, shoes) instead of a shared number, in two places. (a) OWLv2 auto-boxing / fill-in:
  keys get a lower cut-off than the others if the contact sheet shows keys boxes are being lost. (b) The trained detector: the score above
  which a box is shown, chosen from the validation photos by sweeping 0.05 ... 0.95 and taking the value with the best F1 (balance of missed
  vs. false boxes) for each object. Roboflow's guidance says the same: choose the operating point from the precision-recall curve per class and
  confirm on a held-out set; mAP alone does not tell you the cut-off.
- How: `autobox.py`: `THRESHOLD` and `FILL_THRESHOLD` become dicts keyed by prompt label (default value = today's number for every object, so
  behaviour is unchanged until changed); CLI `--threshold` still sets all. `detector.py`: `best_thresholds(model, val_samples)` returns
  `{class: score}` by F1 sweep (greedy IoU 0.5 matching, same as `mean_ap`); skip an object with fewer than 20 validation boxes (keep 0.5);
  save the dict in the checkpoint as `thresholds`; `detect`/`detect_tta` use `min_score=None` meaning "use the saved per-object cut-off, else 0.5".
  The report prints precision, recall and chosen cut-off per object, with the 23f range.
- Acceptance: CPU test with a fake model: an object whose true boxes score about 0.3 gets a cut-off near 0.3 and one whose false boxes score 0.6
  gets above 0.6; old checkpoints without `thresholds` still load and behave as before; fewer than 20 val boxes keeps 0.5. The cut-offs are fitted on
  validation photos only; test and the Phase 2 gold set (task 25) are never used to pick them.
- Expected gain: modest and mostly for keys (more of them found, shown, or labelled); no change to mAP, which ignores the cut-off. Effort: about 50 lines.
  Risk: with few validation boxes the choice is noisy, hence the 20-box minimum and the 0.5 fallback. Licence: none (own code).

### Considered but not recommended
- **Per-object co-teaching for VLM pseudo-labels (arXiv 2511.09955, Nov 2025):** two detectors train together and each drops boxes its partner finds high-loss; +15 mAP over plain pseudo-labels on KITTI. It is built on YOLO (AGPL) and needs two models with per-box losses, which torchvision's Faster R-CNN does not expose. Too much work for our few hundred auto-boxes; the agreement and consistency rules in tasks 23c / 26 are the cheap version.
- **Box-tightness / localisation-quality filtering of OWLv2 boxes:** found no 2025-26 paper specific to it; OWLv2's own authors argue for weak filtering. Revisit if the contact sheets show loose boxes.
- **Noisy tiny-object methods (DN-TOD arXiv 2401.08056, Point Teacher arXiv 2412.05837):** aimed at aerial images, with code not released or not confirmed.
- **Phone-specific depth fine-tuning:** found no 2026 source; the leave-one-out scale fix and the blend in 23l already cover calibration. Apple's Core ML build of Depth Anything V2 Small matters only for on-phone use later (phase 3).

### For Alex to decide
- Nothing new. Still open: RF-DETR / DEIMv2 (10-16) and Depth Anything V3 Small vs V2 Small (10-19).

### Sources
- Open-vocabulary detectors have uncalibrated scores, best cut-off differs by model (Apr 2026): https://arxiv.org/pdf/2604.09920
- How to set a confidence threshold for object detection (Roboflow): https://blog.roboflow.com/how-to-set-a-confidence-threshold-for-object-detection/
- Per-object co-teaching with VLM pseudo-labels: https://arxiv.org/abs/2511.09955
- Scaling open-vocabulary detection (OWLv2, weak filtering of pseudo-labels): https://arxiv.org/abs/2306.09683
- DN-TOD, tiny objects under label noise: https://www.arxiv.org/abs/2401.08056
- Point Teacher: https://arxiv.org/abs/2412.05837
- Depth Anything V2 repo (metric Small models): https://github.com/DepthAnything/Depth-Anything-V2

## 2026-10-25

**Summary.** Started from PR #28 (newest open branch; it includes #25, #26 and #27). Searched for class-imbalance fixes in detection,
fine-tuning schedules for tiny datasets and metric-depth calibration. Nothing changes the detector or depth model choices.
One idea is worth adding for the "how far away" part: right now the number shown in metres is the plain average of two guesses (the depth
model and the object-size guess). Recent robot-navigation papers show that metric depth models can be off by metres in a new room unless
their scale is corrected, so the way the two guesses are combined should be chosen from Alex's tape-measured photos rather than fixed in advance.

### Recommendations

**1. Choose how the two distance guesses are combined from the tape-measured photos (new task 23l, builds on done task 21; small effort, low risk).**
- What: a 2026 robot study (arXiv 2603.19602) found a scale-corrected Depth Anything V2 Small gave 0.69 m mean error while the
  off-the-shelf indoor metric Small model gave 4.19 m, blamed on camera and room differences. An evaluation across six indoor sets
  (CSIR, 2025) found errors grow with distance (89% of errors within 0.5 m up to 2 m, 71% at 6 m). So the metric model's scale is
  the weak point and a phone camera is not what it was trained on. The size-prior guess has the opposite weakness (needs EXIF,
  and real object sizes vary). Averaging two numbers with different error habits is a reasonable default but not necessarily the best one.
- How: in `distance.py` `check_distances`, besides `depth`, `size`, `shown`, also score (a) the geometric mean `sqrt(depth*size)`
  (errors in distance are multiplicative, so averaging in log space treats "too near" and "too far" fairly) and (b) a weighted
  log-space blend `exp(w*log(size) + (1-w)*log(depth))` with `w` fitted leave-one-out on the `distance/` photos (grid 0, 0.1 ... 1).
  Print median % error for each (with the existing leave-one-out scale fix) and make the winner the number `estimate_distances`
  shows by default (a module constant `BLEND_W`, default 0.5 = geometric mean until real data exist). Also: when one photo has two or more
  detections with a size prior (bottle/shoe, not keys), fit ONE scale for the photo, `median(size_m / depth_m)`, apply it to the
  depth-model metres of all objects in that photo (so keys get a calibrated number too) and flag `calibrated_by_photo=True`.
- Acceptance: test with fake maps: geometric mean of 1 m and 4 m is 2 m; the blend weight picks 1.0 when the depth guess is wildly wrong and 0.0
  when the size guess is; the per-photo scale makes a keys box in a photo with a bottle land at the right distance when the metric map is
  uniformly 2x too large. Report keeps saying near/far order is reliable and metres approximate.
- Expected gain: modest for metres (guess: a few points of % error), none for near/far order. Effort: about 40 lines. Risk: low; with few
  `distance/` photos the fitted `w` is noisy, hence the default of 0.5 until there are 10+ photos. Licence: none (own code).

### Considered but not recommended
- **Repeat-factor sampling / E-IRFS for class imbalance (arXiv 2403.07113, 2503.21893):** the downloader already caps every object at ~300 photos, so classes are balanced; revisit only if Alex's own footage skews one class.
- **Learned per-image depth calibration heads (arXiv 2601.01457):** trains new heads and needs captions; too much for the gain when a size prior is available.
- **Progressive fine-tuning with plateau-aware LR (CVPR 2026, arXiv 2603.28182):** aimed at 1-10 shot cross-domain detection; the current warm-up + cosine + early stopping recipe (23a) is adequate at hundreds of boxes. I could not read its supplement.
- **YOLO26-Depth:** Ultralytics, AGPL code.

### For Alex to decide
- Nothing new. Still open: RF-DETR / DEIMv2 (10-16) and Depth Anything V3 Small vs V2 Small (10-19).

### Sources
- Robot navigation with scale-corrected Depth Anything V2 (0.69 m vs 4.19 m): https://arxiv.org/pdf/2603.19602
- Indoor metric depth error versus distance (2025 evaluation): https://researchspace.csir.co.za/signposting/describedby/28469156-78e2-4323-85c8-ed7e689b9378
- Language as Prior, Vision as Calibration (per-image affine in inverse depth): https://arxiv.org/html/2601.01457v1
- Class imbalance in object detection study: https://arxiv.org/pdf/2403.07113
- Cross-domain few-shot detection, fine-tuning matters (CVPR 2026): https://arxiv.org/abs/2603.28182
- Depth Anything V2 metric indoor small: https://huggingface.co/kerasformers/depth_anything_v2_metric_indoor_small

## 2026-10-24

**Summary.** Started from PR #27 (newest open branch; it includes #25 and #26). Searched for new detector families and licences, open-vocabulary
labellers, small metric-depth models, frame-sampling practice and training-stability tricks. Nothing changes the detector, depth or
auto-labelling choices. One small, cheap training idea is worth a trial: keep a smoothed ("moving average") copy of the detector's
weights while training. It costs a few lines, uses only PyTorch, and may help most when the tiny validation set makes the
per-epoch score jumpy. The evidence for it is indirect (no clean small-data study found), so it is an ablation, not a promise.

### Recommendations

**1. Weight moving average (EMA) as a training option (new task 23k; tiny effort, low risk).**
- What: keep a second copy of the weights that is updated after every optimiser step as `ema = d*ema + (1-d)*weights`.
  The smoothed copy is usually a little better and much steadier than the raw weights, which matters when training uses only
  hundreds of photos and a few dozen validation photos (best-epoch choice is noisy). Semi-supervised and few-annotation
  detection papers (2023-2025) use an EMA with a cosine-scheduled keep rate and report about +0.7 point from the schedule; popular
  YOLO/RT-DETR/D-FINE trainers switch EMA on by default. No study I found isolates its gain for plain fine-tuning on a
  small set, so keep it only if validation shows it helps.
- How: in `train_detector` (detector.py) add `ema_decay=0.0` (flag `--ema-decay`, 0 = off). Use
  `torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(decay), use_buffers=True)` and call
  `ema.update_parameters(model)` after `sched.step()`. Try 0.999 (about 1000 steps memory; with ~15 epochs of a few hundred
  steps lower it to 0.99-0.995 if the run is short). Score validation mAP on `ema.module` instead of `model`, and keep
  best-epoch weights from the EMA copy. Save the EMA weights in the checkpoint. torchvision detection models use frozen
  BatchNorm, so there are no running statistics to worry about. Skip the first warm-up iterations
  (copy weights until iteration `warmup`, then start averaging).
- Acceptance: CPU test with `weights=None`: after a few steps with decay 0.9 the EMA weights differ from the raw weights and
  lie between the start and current values; decay 0 leaves behaviour unchanged (existing tests untouched). Ablation
  `--ema-decay` 0 / 0.995 on the validation set with the 23f range; keep it on only if overall mAP does not drop and the
  best-epoch curve is steadier.
- Expected gain: small (guess 0-2 mAP, mostly steadier results). Effort: about 15 lines. Risk: too slow a decay under-trains a
  short run, hence the ablation. Licence: PyTorch (BSD-3), no new library.

### Considered but not recommended
- **SAM 3 for auto-boxing instead of OWLv2:** Meta's custom "SAM License" is not Apache-2.0 (limits on redistribution, export-control and military uses, can be changed by Meta); commercial suitability unclear and the model is much larger. Stay with OWLv2 (Apache-2.0).
- **DEIMv2 / D-FINE:** the Atto-to-Nano sizes use HGNetv2 backbones, but the S-to-X sizes use DINOv3 backbones under the separate DINOv3 licence, and community repackages (e.g. OV-DEIM) can be CC BY-NC. Not worth the swap before the first real results; still on "For Alex to decide".
- **Frame difference / uniform frame sampling for training (arXiv 2505.18302, 2025):** supports dropping near-identical video frames, but I could only read the abstract (the sandbox cannot fetch arXiv). Task 24 already plans perceptual-hash de-duplication for walks; revisit for the 2 fps training frames only if the own-footage score looks inflated after 23h.
- **Depth Pro / UniDepthV2 / Depth Anything 3 metric Large:** licences or sizes still unconfirmed or too large (same conclusion as 10-16 to 10-21).
- **More unfreezing / bigger augmentation recipes:** covered 10-20; the far-object work (23g tiles, 23j zoom windows) is the right next experiment.

### For Alex to decide
- Nothing new. Still open: RF-DETR / DEIMv2 (10-16) and Depth Anything V3 Small vs V2 Small (10-19).

### Sources
- Few-annotation detection study (EMA with cosine keep-rate schedule): https://arxiv.org/pdf/2310.19936
- Training-based model refinement for semi-supervised detection: https://arxiv.org/pdf/2307.13755
- Bag of tricks for real-time detection (EMA at several decays): https://arxiv.org/pdf/2508.19804
- Two-stage training paradigm for small datasets: https://arxiv.org/pdf/2309.05652
- Sampling strategies for efficient detector training (abstract only): https://arxiv.org/pdf/2505.18302
- SAM 3 licence: https://huggingface.co/facebook/sam3/blob/main/LICENSE
- DEIMv2 (Apache-2.0 code, DINOv3 backbones): https://arxiv.org/abs/2509.20787
- OV-DEIM weights repackaging, CC BY-NC 4.0: https://huggingface.co/LibreYOLO/LibreOVDEIMs

## 2026-10-22

**Summary.** Started from PR #25 (bigger photos, 15 rounds), the newest open branch. That PR exists because run 3 found 87% of
near objects but only 28% of far ones. This run looked for ways to help small, far objects beyond making the photo bigger.
Nothing changes the detector or depth choices. The one useful, well-supported idea is "look closer": run the detector on
zoomed-in overlapping pieces (tiles) of the photo as well as the whole photo, and train on zoomed windows so that
training and test objects look the same size. Tiling has a lot of published support for small objects, costs no new
training data, and needs only a small extra library we already planned (`ensemble-boxes`).

### Recommendations

**1. Add tiles to test-time augmentation (refines task 23g; small effort, low risk).**
- What: SAHI (Slicing Aided Hyper Inference) cuts a photo into overlapping tiles, runs the detector on each tile and
  merges the boxes. The original paper reports +5 to +7 AP on small objects from inference alone, with no retraining
  of the detector. Newer work (2026) says the weak point is how tile results are merged and ranked, and that tiling
  wastes time on sparse scenes; our photos have 1-3 objects and the tiles are only 4, so cost is small.
- How: extend the planned `detect_tta` (23g) with a `tiles` option: full photo + 4 tiles (2x2, each 60% of the side,
  ~20% overlap, resized to the model size), boxes mapped back, merged by `weighted_boxes_fusion` (iou_thr 0.55). I
  suggest writing the tiling ourselves (about 30 lines) rather than installing `sahi`: it keeps Colab setup simple and
  works directly with our torchvision model.
- Expected gain: the best-supported fix for the far bin (28%); a guess of +5-10 points far recall. Risk: duplicate or
  cut-off boxes at tile edges (WBF handles most). Cost: 5x inference time, so off by default and off in the web page.
  Licence: `ensemble-boxes` MIT; SAHI itself MIT (not used).

**2. Zoomed-window training crops (new task 23j; small effort, low risk).**
- What: Roboflow (2026) and the MathWorks tiled-training guide both say tiling works best when training and inference
  see objects at the same scale. Training photos are shown whole (with affine zoom 0.5-1.5), but a tile at test time
  zooms objects by about 1.7x, a case training rarely shows.
- How: see task 23j (random 50-70% window that keeps at least one whole box, p=0.25, behind a flag, ablation).
- Expected gain: makes recommendation 1 work better and also gives far objects more training pixels. Risk: windows can
  drop context; mitigated by the flag and the keep-only-if-helps rule. Licence: own code.

### Considered but not recommended
- **Smaller anchors (the torchvision MobileNet model uses sizes 32-512 on coarse feature maps):** plausible, but our "far"
  bin is under 3% of the photo area (roughly up to 130 px at 640x960), which the current anchors cover; changing them
  also forces new RPN weights and loses the COCO-pretrained head. Revisit only if tiles do not help.
- **Full tiled training at native resolution (MathWorks YOLOX style):** needs photos larger than the 960 px we store; the zoomed windows in 23j get most of the effect.
- **ROI-gated / content-adaptive SAHI (arXiv 2608.23923):** aimed at big sparse scenes (drones); our images are small.
- **Training-free tile re-scoring preprint (Research Square):** unreviewed, one dataset; WBF is enough for now.

### For Alex to decide
- Nothing new. Still open: RF-DETR / DEIMv2 (10-16) and Depth Anything V3 Small vs V2 Small (10-19).

### Sources
- Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection (SAHI): https://arxiv.org/abs/2202.06934
- ROI-Gated SAHI (content-adaptive slicing): https://arxiv.org/pdf/2608.23923
- Edge tiling during inference (Roboflow): https://roboflow.com/blog/edge-tiling-during-inference
- Detect small objects using tiled training (MathWorks): https://www.mathworks.com/help/vision/ug/detect-small-objects-using-tiled-training-yolox.html
- Cascaded zoom-in detector: https://www.arxiv.org/pdf/2303.08747
- ISPRS 2026 tiling speed-up for SAHI-style pipelines: https://isprs-annals.copernicus.org/articles/XI-3-2026/81/2026/isprs-annals-XI-3-2026-81-2026.pdf
- torchvision Faster R-CNN MobileNetV3 source (anchor sizes): https://github.com/pytorch/vision/blob/main/torchvision/models/detection/faster_rcnn.py

## 2026-10-21

**Summary.** Started from PR #19 (which includes #18 and #12-#17). Searched for new small-data detectors, small
metric-depth models, per-class threshold tuning and self-training with noisy pseudo-boxes. Nothing changes the
detector or depth choices: the Apache-2.0 detector field is still RF-DETR, D-FINE and DEIM(v2), and Depth Anything V2 Small
(plus V3 Metric, still Large only) is unchanged. One idea is worth adding for the later "robot walks and retrains itself"
phase: self-labels are more trustworthy when the same object is seen consistently over several neighbouring video frames.

### Recommendations

**1. Frame-to-frame consistency check for self-labels (task 26, small effort, low risk).**
- What: "Look Around and Learn" (exploration self-training, 2023) and the 2026 box-level-surrogate work
  both report that a detector trained on its own outputs drifts (confirmation bias) unless labels are cross-checked,
  and that confidence thresholds alone do not remove noisy boxes. Walk videos give a free cross-check: a real object
  persists across neighbouring frames, a false alarm usually flickers.
- How: in the auto-labelling step (task 26), after the detector + OWLv2 agreement test, keep a box only if a box of the
  same label with IoU > 0.3 (after allowing for small motion) exists in at least 2 of the 4 nearest sampled frames of the
  same walk (pure numpy IoU matching, no tracker library). Boxes that fail go to the review batch only if confidence is
  high; otherwise they are dropped. Log how many boxes each rule removed.
- Expected gain: fewer wrong self-labels entering training, so the quality gate (task 27) rejects fewer candidates.
  Effort: about 40 lines plus a test with a fake 5-frame walk (one steady box, one that appears once). Risk: low; very fast
  motion may drop true boxes, which only costs a few labels. Licence: none (own code).

### Considered but not recommended
- **Per-class confidence thresholds chosen from F1 sweeps:** sensible, but with a few hundred test photos per object the choice is noisy; revisit once the gold set (task 25) exists.
- **Ultralytics YOLO26 depth models:** licence not confirmed (Ultralytics is AGPL for code), skip.
- **Depth Anything 3 Metric Large:** Apache-2.0 but Large (0.35B), too heavy for a free Colab alongside the detector.
- **Co-teaching two YOLO models to filter noisy boxes (arXiv 2511.09955):** YOLO is AGPL; the idea overlaps with the OWLv2 + detector agreement rule.
- **Metasurface depth (ECCV 2026):** needs special encoded image pairs, not ordinary photos.

### For Alex to decide
- Nothing new. Still open: RF-DETR / DEIMv2 (10-16) and Depth Anything V3 Small vs V2 Small (10-19).

### Sources
- Look Around and Learn: Self-Training Object Detection by Exploration: https://arxiv.org/pdf/2302.03566
- Pseudo-label Correction and Learning for semi-supervised detection: https://ar5iv.labs.arxiv.org/html/2303.02998
- Co-teaching to filter noisy boxes: https://papers.cool/arxiv/2511.09955
- How to set a confidence threshold for object detection (Roboflow): https://blog.roboflow.com/how-to-set-a-confidence-threshold-for-object-detection/
- DEIM (Apache-2.0): https://github.com/Intellindust-AI-Lab/DEIM
- Depth Anything 3 metric Large: https://huggingface.co/APaul1/DA3METRIC-LARGE

## 2026-10-20

**Summary.** Started from PR #18 (which includes #12-#17). No new model or licence news: searches for small-data
detector fine-tuning and small metric-depth models found nothing newer than what is already in this log (Depth
Anything V3 metric is still Large-only; V2 Small stays the right size). One useful finding about evaluation: when
frames come from video, a split "by video" still leaks if several of Alex's videos show the same room, table or
shelf, because the backgrounds match. Recent papers (2025-2026) show this inflates scores. A cheap fix is to
keep whole look-alike *groups of videos* on one side of the split.

### Recommendations

**1. Scene-aware train/test split (small effort, low risk).**
- What: Glazner et al. ("Find the Leak, Fix the Split", 2025) and a 2026 driving-perception study show that
  neighbouring or look-alike frames on both sides of a split make mAP look better than it is, and that exact-file or
  per-video checks do not catch it. Here, Alex films each object in a few places, probably the same few rooms, so a
  "different video" is often the same scene.
- How: in `detection_data.py` add `scene_groups(samples, threshold)`: for each video/photo group compute one small
  descriptor of its background (mean of a 16x16 downscaled RGB thumbnail of its frames, or a 3x8-bin colour histogram;
  numpy/PIL only, no model download), then merge groups whose descriptors are closer than `threshold` (single-linkage;
  `scipy.cluster.hierarchy` or a simple union-find). Make `split_detection` split by these merged scene groups (Open Images
  photos stay one group each). Print how many own videos merged into how many scenes. If everything merges into a
  single scene, fall back to the old by-video split and say so.
- Acceptance: test with fake videos (two with the same background colour, one different): the two look-alikes always land on the
  same side. The report states the number of test scenes next to each score; with fewer than 3 own test scenes, say the own-footage numbers
  are rough.
- Expected gain: no change in the model, but honest scores, which matters for the "keep only if it helps" calls
  (23b, 23d, 23e, 23g) and the phase-2 gate (task 27). Licence: none needed.

### Considered but not recommended
- **Heavier fine-tuning tricks (self-supervised pre-training, P2 high-resolution head, mosaic/mix-up from 2023 two-stage paper):** needs far more data or compute than a free Colab session; mosaic/mix-up overlap with 23b.
- **Unfreezing more backbone layers:** a Faster R-CNN transfer-learning study says the benefit shrinks with less data; the current mostly-frozen setup is right.
- **Depth Pro / UniDepthV2 for metres:** metric and interesting, but licences and small-size variants not confirmed; V2 Small plus the size-prior estimate (task 21) is enough until real distance results exist.
- **Weight averaging (EMA) of the detector:** no source found in this search; revisit only if validation mAP is jumpy.

### For Alex to decide
- Nothing new. Still open: Depth Anything V3 Small vs V2 Small (10-19), RF-DETR / DEIMv2 (10-16).

### Sources
- Find the Leak, Fix the Split (cluster-based splitting for video-derived data): https://arxiv.org/html/2511.13944v2
- Leakage in KITTI (Volvo-linked study): https://arxiv.org/html/2410.23312v1
- Domain shift between videos and images (few frames per shot, many videos): https://arxiv.org/pdf/1501.01186
- Impact of dataset size and fine-tuning on Faster R-CNN: https://dspace.ut.ee/items/0ee00175-ba34-4ae5-8944-b4cee9c8b680/full
- Two-stage training paradigm for small datasets: https://arxiv.org/pdf/2309.05652
- Depth Anything V2 metric indoor small (Apache-2.0): https://huggingface.co/kerasformers/depth_anything_v2_metric_indoor_small

## 2026-10-19

**Summary.** Started from PR #16 (which includes #12-#15). Another quiet day for models: no new small
detector beats the current choice, and for few-shot fine-tuning the old advice still stands (freeze most of the
network when data is tiny, which the code mostly does). One new, cheap idea: at prediction time, look at each photo
a few ways (as is, mirrored, slightly larger) and merge the boxes. It needs no retraining, directly serves
"any angle / position", and costs only a few extra seconds per photo. Also checked Depth Anything V3 (new),
whose licences differ by size; see "For Alex to decide".

### Recommendations

**1. Test-time augmentation with box merging (small effort, low risk).**
- What: run the detector on the photo, its left-right mirror, and a ~1.3x upscaled copy; map the mirrored/scaled
  boxes back to the original photo; merge overlapping boxes of the same label with Weighted Boxes Fusion (WBF),
  which averages boxes instead of discarding all but one as NMS does. Published results (small-object and
  competition detection, Open Images and COCO for WBF itself) show gains, mostly for small or partly hidden objects.
  No study I found measures it on a small training set, so treat the gain as unproven: hence the "keep only if it helps" rule.
- How: in `detector.py` add `detect_tta(model, image, classes, min_score, flips=True, scales=(1.0, 1.3))`
  using the `ensemble-boxes` package (`weighted_boxes_fusion`, boxes normalised to 0-1, `iou_thr=0.55`,
  `skip_box_thr=0.05`, `conf_type="avg"`). Un-flip x coordinates (x1' = 1 - x2, x2' = 1 - x1) before fusing; this
  is the usual bug. Keep `detect` as the default; add `--tta` to the evaluation CLI and the notebook photo demo.
- Acceptance: unit test that a mirrored fake detector's box is mapped back exactly; evaluation prints mAP with and without
  TTA with the confidence range from 23f, and TTA stays off unless it helps overall without hurting any object.
- Expected gain: modest (guess: +1-3 mAP on small/edge objects). Cost: 3x inference time, which only matters for the web page
  (task 23: keep it off there) and phase 3 live use. Licence: `ensemble-boxes` is MIT.

### Considered but not recommended
- **Staged unfreezing / training only the last layers (TFA-style few-shot recipe):** the dataset here (hundreds to thousands of boxes) is above the few-shot regime; revisit only if 23a shows overfitting.
- **Extreme R-CNN style synthetic samples and distillation:** the idea overlaps with 23b (composite pasting); the distillation part is too much work.
- **Depth Anything V3 metric:** only a Large metric model exists (Apache-2.0, too big for a free GPU alongside the detector); no small metric one.
- **YOLO-family TTA tutorials:** AGPL licence, no.

### For Alex to decide
- **Depth Anything V3 Small / Base (Apache-2.0, newer than the V2 Small you chose):** its Large and Giant versions are non-commercial, but Small and Base are Apache-2.0 and reported more accurate. Swapping is your model choice, so I changed nothing. Suggested path if you like it: after the first real distance results (task 21) are in, compare V2 Small and V3 Small on your `distance/` photos. Still open: RF-DETR / DEIMv2 from 10-16.

### Sources
- Weighted Boxes Fusion (Open Images / COCO): https://arxiv.org/abs/1910.13302
- Small bird detection with TTA + WBF: https://arxiv.org/abs/2401.01018
- Annotation-free spacecraft detection (WBF vs NMS discussion): https://arxiv.org/abs/2602.04699
- Frustratingly Simple Few-Shot Object Detection: https://arxiv.org/abs/2003.06957
- Extreme R-CNN: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11645051/
- Depth Anything 3 licences by model size: https://github.com/ByteDance-Seed/Depth-Anything-3

## 2026-10-18

**Summary.** Started from PR #15 (which already includes #14, #13 and #12). No better detector or
depth model has appeared, so no model change. Three things came out of reading `detector.py` and
testing the web-page plan: (1) the current "rotate up to 20 degrees" augmentation draws loose boxes
that, per a published study, can make results *worse* than no rotation, so "any angle" should use
exact 90-degree turns plus a small tilt; (2) the plan for the web page (task 23) can be made concrete:
the detector exports to ONNX in the Claude sandbox, and the depth model has a ready-made browser
version; (3) with only a few held-out videos, differences of 1-2 mAP points are noise, so the
"keep it only if it helps" decisions (23b, 23d, 23e, and the phase-2 gate) need a confidence range.

### Recommendations

**1. Fix the rotation augmentation: exact 90-degree turns, smaller tilt, and an ablation (small effort, low risk).**
- What: `train_transform` uses `RandomAffine(degrees=20)`. A rotated box is re-drawn as the box around
  the rotated corners, which grows with the angle and includes background. Kalra et al. (ICCV 2021)
  report this "largest box" labelling is only safe below about 3 degrees and often does worse than no
  rotation; Shibata et al. (ACCV 2022) say the same. For exact 90-degree turns the box is computed
  exactly, so there is no such loss.
- How: in `detector.py` set `RandomAffine(degrees=8, ...)` (keep translate and scale), and add
  `v2.RandomApply([v2.RandomChoice([v2.RandomRotation((90, 90), expand=True), v2.RandomRotation((270, 270), expand=True)])], p=0.15)`
  plus `v2.RandomVerticalFlip(p=0.1)` before it (torchvision v2 rotates `BoundingBoxes` exactly for these angles).
  A bottle lying on its side and a shoe upside down are the realistic "any angle" cases for a phone camera.
  Expose `--rot-degrees` and `--rot90-p` in the training CLI.
- Acceptance: unit test that a 90-degree rotation of a known box lands on the exact expected pixels;
  ablation on the same split: degrees 0 / 8 / 20, with and without 90-degree turns. Keep the setting with the best
  validation mAP (see rec. 3 for how to judge a difference); report the result in PROGRESS.md.
- Expected gain: uncertain but plausible on lying/rotated objects (it is a label-quality fix, not a model change).
  Risk: Open Images photos are mostly upright, so keep p small. Licence: torchvision (BSD).

**2. Make the web page plan (task 23) concrete (small effort, medium risk).**
- Tested here (Claude sandbox, torch 2.14, random weights): `torch.onnx.export(model, ([x],), "d.onnx", opset_version=17, dynamo=False)`
  for the current Faster R-CNN MobileNetV3 works with a fixed 3x480x640 input, loads in `onnxruntime`, and outputs
  boxes/labels/scores. File size is about 76 MB in float32. I did not confirm that outputs match PyTorch (random-weight
  scores are all near-equal, so a comparison of the top boxes was meaningless); this must be checked with the trained
  checkpoint.
- How: `claudeknows/export_web.py` gets `export_detector_onnx(checkpoint, out)` (same call, `dynamo=False`, opset 17) and a test
  that exports a tiny `weights=None` model and loads it with `onnxruntime` (add `onnx` and `onnxruntime` to a dev-only
  requirements file). Acceptance on a trained checkpoint: the same photo gives the same boxes (within 1 px, same labels)
  for scores above 0.5 in PyTorch and ONNX. In the browser: `onnxruntime-web` (MIT, WASM backend) for the detector, letterbox the
  photo to 640x480; depth via Transformers.js `depth-estimation` pipeline with `onnx-community/depth-anything-v2-small`
  (Apache-2.0). The size-prior distance (task 21) needs no model and runs in plain JS.
- Keep the fall-back already in the task: if parity fails or the page is too slow, link to the notebook instead. Host the
  76 MB model as a GitHub Release asset, not in git history (or shrink with `onnxruntime` float16/int8 quantisation if parity holds).
- Expected gain: Alex can try the model from a phone browser without Colab. Risk: medium (post-processing ops in ONNX
  are the usual trouble spot; an NVIDIA forum report hit a TensorRT shape error on this very model, which is a different runtime).

**3. Report a confidence range for every "did it help?" number (small effort, low risk).**
- What: held-out sets here are a handful of videos. A change of one or two mAP points can come from which frames
  landed in the test set. The standard fix is a bootstrap that resamples whole videos (frames of one video are not independent).
- How: in `build_report` add `bootstrap_ci(results, groups, n=1000, seed=0)`: resample test *groups* (video or
  photo) with replacement, recompute mAP@0.5 per object and overall, report the 2.5-97.5% range next to each number. numpy only.
  For A/B decisions (23b composite, 23d negatives, rec. 1 above) compute the paired difference on the same resamples and say
  "helped / no clear difference / hurt". Phase 2's gate (task 27: "no object worse by more than ~2 points") should use the same range.
- Acceptance: test with a tiny fake result list where the range is wider with 3 groups than with 30. Expected gain: avoids
  keeping or dropping features based on noise. Licence: none needed.

### Considered but not recommended
- **Initialising the water-bottle head from COCO's "bottle" row:** cute, but bottles are the object with the most data; little to gain.
- **Freezing more of the backbone / staged unfreezing for the detector:** forum advice only, nothing authoritative found for this size; revisit if validation mAP (23a) shows overfitting.
- **Rotation-uncertainty loss and augmented feature pooling (Kalra, Shibata):** research code, more work than the 90-degree fix.
- **A 2026 paper on small-data detector fine-tuning with OWLv2 pseudo-labels:** searched, found none; the OWL-ST and AIDE papers confirm raw OWLv2 labels are noisy, which fits the existing 0.4 cut-off in 23c.
- **YOLO in the browser (YOLOv8/10/11 ONNX demos are common):** AGPL licence (Ultralytics), so no.

### For Alex to decide
- Nothing new. (Still open: the RF-DETR / DEIMv2 question from 10-16.)

### Sources
- Kalra et al., Towards Rotation Invariance in Object Detection, ICCV 2021: https://arxiv.org/abs/2109.13488
- Shibata et al., Robustizing object detection with augmented feature pooling, ACCV 2022: https://openaccess.thecvf.com/content/ACCV2022/papers/Shibata_Robustizing_Object_Detection_Networks_Using_Augmented_Feature_Pooling_ACCV_2022_paper.pdf
- Depth Anything V2 Small ONNX for Transformers.js (Apache-2.0): https://huggingface.co/onnx-community/depth-anything-v2-small
- Running ONNX models in the browser with onnxruntime-web: https://pyimagesearch.com/2025/07/28/run-yolo-model-in-the-browser-with-onnx-webassembly-and-next-js/
- torchvision Faster R-CNN ONNX export discussion: https://forums.developer.nvidia.com/t/issues-with-onnx-to-tensorrt-conversion-for-the-faster-r-cnn-mobilenet-v3-model/256202
- OWL-ST (pseudo-label noise): https://arxiv.org/html/2306.09683v3

## 2026-10-17

**Summary.** Short follow-up. Started from PR #14 (which already includes PR #12 and the 10-16
research PR #13). A fresh search for newer detectors, depth models and pseudo-label methods
turned up nothing that beats the current choices (Faster R-CNN MobileNet, OWLv2, Depth Anything
V2 Small): the Apache-2.0 detector field is still RF-DETR, D-FINE and DEIM(v2), all already
considered or equivalent, and the small Depth Anything V2 metric (Hypersim) checkpoint is still
the only one of its metric family that is Apache-2.0. Reading `autobox.py` and `train_detector`
did show two cheap gaps in how the *labels* are built, which matter more than a new model for
a dataset this small. Both are label fixes, not model changes.

### Recommendations

**1. Fill in the missing boxes in single-object folders (small effort, low risk).**
- What: today `autobox_folder` keeps only the folder's own object (`expect`), and Open Images
  water-bottle / shoe photos only carry human boxes for that one class. So a shoe standing next
  to a bottle in the "water_bottle" folder is taught to the detector as *background*, which
  directly fights the goal "several objects per photo".
- How: in `autobox.py` add a second pass, `fill_missing(folder, detect, threshold=0.4)`, run for
  every folder (web and own, including ones already boxed by humans): run OWLv2 with all three
  prompts; for any detection of a class *other than* what the photo already has, with score >=
  0.4 and IoU < 0.3 against every existing box, append it with `by="owlv2-extra"`. Never touch
  or remove human boxes. Keep the stricter 0.4 (vs 0.3 elsewhere) because a wrong extra box
  teaches a false positive. Include these boxes in contact sheets (`only_by` already filters by
  source) so Alex can glance at them.
- Acceptance: test with the fake detector that extra boxes are added only for other classes,
  never overlap existing boxes, and human boxes stay byte-identical; the report prints how many
  extra boxes were added per folder.
- Expected gain: fewer "missed bottle/shoe" and cleaner multi-object behaviour; complements the
  copy-paste idea (23b), which adds *synthetic* multi-object scenes. Risk: OWLv2 errors on
  keys-like clutter; the 0.4 threshold and the by-tag (easy to drop later) limit that.
  Licence: OWLv2 Apache-2.0, no new dependency.
- Support: cross-detector agreement is the standard way to filter pseudo-labels, and the
  closest study found 5.5% false auto-labels without filtering (see sources); an
  "ignore" state for unverified objects is also how Open Images' own challenge scores it.

**2. Add background-only (negative) photos (small effort, low risk).**
- What: torchvision detectors accept images with zero boxes (`boxes` of shape `(0, 4)`,
  `labels` of shape `(0,)`), but the training data currently has only photos that contain at
  least one object. The detector never learns "this is none of my three things", the same
  weakness the classifier showed (a cat came out as "shoes, 62.5%").
- How: (a) download ~100-150 Open Images photos that are human-verified to contain none of
  the targets: in the image-level label file the rows ending `,0.0` for a class are verified
  absences, so take image IDs verified-absent for ALL of water bottle, key, keychain, shoe and
  footwear (look up the footwear MID in the class-descriptions file); prefer ones that are
  verified positive for indoor things (e.g. table, chair, floor, person) so they look like Alex's
  setting. (b) Alex's own frames with no objects (a `claudeknows_photos/empty/` folder is
  optional, not required; just skip it if missing). (c) In `DetectionDataset` allow
  empty-box samples; cap negatives at ~15% of each epoch's samples. (d) Add to `build_report`
  "false alarms per negative photo at score 0.5", and report it next to mAP.
- Acceptance: a CPU test that one training step with an all-negative batch runs, and that the
  report shows the false-alarm number. Never include negatives in the gold/test positives.
- Expected gain: fewer false detections on random photos, which is what a walking robot sees
  most of the time. Risk: low; too many negatives can lower recall, hence the 15% cap.
  Licence: same Open Images (CC BY) credits as the rest.

### Considered but not recommended
- **DEIMv2 / D-FINE (Apache-2.0)**: strong COCO numbers, but no small-data fine-tuning
  comparison found; same answer as RF-DETR yesterday (Alex's call, see below).
- **Label-Verify-Correct / OWL-ST style self-training for the detector**: the full methods are
  built for thousands of images; the simple version (agreement + extra boxes) is rec. 1 and
  the phase-2 loop (task 26) already covers agreement.
- **Loss-based pseudo-label filtering (Gaussian mixture, PLS)**: classification method, adds
  complexity for few hundred boxes; revisit in phase 2.
- **Self-supervised "detector teaches itself" (May 2026 preprint)**: aimed at adapting
  vision-language models, not at small detectors.

### For Alex to decide
- Nothing new. The RF-DETR / DEIMv2 question from 10-16 still stands (only after the first
  real detector scores are in).

### Sources
- Open Images V4 paper (positive and negative image-level labels): https://arxiv.org/pdf/1811.00982
- Open Images challenge 2019 detection (unannotated classes are ignored): https://www.tensorflow.org/datasets/catalog/open_images_challenge2019_detection
- Label, Verify, Correct (few-shot detection with pseudo-labels): https://openaccess.thecvf.com/content/CVPR2022/papers/Kaul_Label_Verify_Correct_A_Simple_Few_Shot_Object_Detection_Method_CVPR_2022_paper.pdf
- OWL-ST, scaling open-vocabulary detection by self-training: https://arxiv.org/html/2306.09683v3
- Auto-labelled detector data with multi-detector voting (thesis, 5.5% false labels): https://www.ifp.uni-stuttgart.de/lehre/masterarbeiten/573-wiedemann/
- PLS, label-noise robustness: https://arxiv.org/pdf/2210.04578
- DEIMv2 (DINOv3 real-time detector): https://arxiv.org/html/2509.20787v4
- Depth Anything V2 licences (Small Apache-2.0): https://github.com/DepthAnything/Depth-Anything-V2 and https://docs.lightly.ai/train/stable/depth_estimation.html

## 2026-10-16

**Summary.** First research pass. Starting from PR #12 (tasks 19-20: OWLv2 auto-boxing and a
Faster R-CNN MobileNetV3 detector). The model choices already made (Faster R-CNN MobileNet,
OWLv2, Depth Anything V2 Small) are sound and permissively licensed, so I am **not**
suggesting any model swap. Reading the code, three gaps stand out that matter more than a
new model: (1) the training loop has no LR warm-up/schedule, no mixed precision and no
held-out check to pick the best epoch; (2) most Open Images photos only box ONE kind of
object, so unboxed shoes/bottles in the same photo are silently taught as "background", and
the detector never sees realistic multi-object scenes except Alex's `together/` videos;
(3) the metres from the metric-indoor depth model will be shaky, and there is a free,
licence-clean second estimate available from the known real-world size of the objects.

### Recommendations

**1. Better training recipe for the detector (small effort, low risk).**
- What: add to `train_detector` in `claudeknows/detector.py`: linear warm-up for ~200
  iterations then cosine decay (`torch.optim.lr_scheduler.LinearLR` + `CosineAnnealingLR`,
  or `OneCycleLR`); mixed precision (`torch.autocast("cuda", dtype=torch.float16)` +
  `torch.cuda.amp.GradScaler`, only when on CUDA, so CPU tests are unchanged); gradient
  clipping (`clip_grad_norm_(params, 10)`); keep batch size 4-8 with SGD lr 0.005 x
  (batch/8) (torchvision's reference recipe scales lr linearly with batch; the current
  1e-3 is on the low side for a new head but fine for the backbone, so warm-up + schedule
  is what matters).
- Why here: with only hundreds of boxes, a flat LR either under-trains the new head or
  wobbles at the end, and without a held-out check there is no way to know whether epoch 10
  was better than epoch 6. Today `split_detection` gives only train/test.
- How: split a **validation** set off the train set (by video/photo, same
  `split_by_group`), compute mAP@0.5 on it after each epoch (reuse `predict_samples` +
  `average_precision`), save the best-epoch weights, stop after 4 epochs with no
  improvement. Test set stays untouched for the final report.
- Expected gain: modest (a few mAP points, mostly stability); the main value is trustworthy
  numbers. Risk: low. Licence: torch only (BSD).

**2. Multi-object composite ("copy-paste") augmentation (medium effort, medium risk).**
- What: Simple Copy-Paste (Ghiasi et al., CVPR 2021) and the older small-object copy-paste
  of Kisantal et al. 2019 show that pasting object instances into other images is a strong
  augmentation for detection, especially for rare/small objects and when few images have
  several objects.
- Why here: Open Images photos in the water-bottle folder only box bottles, in the shoes
  folder only footwear, and keys come from OWLv2 on single-key photos. So the training set
  almost never contains a bottle next to shoes, yet that is exactly what Alex wants
  ("keys bottom-left, bottle centre"). Unlabelled objects in a photo also count as false
  positives during training.
- How: new `composite_transform` in `detector.py` (or a small `claudeknows/composite.py`),
  applied with p~0.3 before the existing box-aware augmentation. Take a random box crop
  (from any training image of another class) and paste it into the current image at a
  random location with random scale (0.5x-1.5x of its original size, capped to <= 40% of
  the target image side), feathered edges (Gaussian-blurred alpha mask, ~5 px) so the
  rectangle border is not a giveaway; reject placements with IoU > 0.1 against any existing
  box; add the new box + label. Paste 1-3 objects. Use only train-split crops (never test or
  gold images). Pure torch/PIL, so no new dependency. Do NOT use the `ultralytics`
  package for its mosaic/copy-paste: it is AGPL-3.0.
- Acceptance: unit test that pasted boxes lie inside the canvas, don't overlap, and labels
  match; report mAP with and without it on the same split (flag in the training CLI,
  `--composite 0.3`), and keep it only if it helps on the `together/` held-out videos.
- Expected gain: uncertain but plausible on multi-object scenes (I found no 2026 result
  specific to datasets this small, so treat it as an experiment). Risk: pasted rectangles
  have no mask, so pasted objects carry some background; feathering limits the damage.

**3. Distance: add a size-prior estimate and calibrate honestly (small effort, low risk).**
- What: refines task 21. Besides Depth Anything V2 (relative + metric-indoor Small), compute
  a **pinhole estimate**: `distance_m = focal_px * real_size_m / box_size_px`, with
  `focal_px = FocalLengthIn35mmFilm / 36 * image_width_px` read from the photo's EXIF
  (Pillow `Image.getexif()`; fall back to the depth model alone if missing, as happens for
  video frames and downloaded photos). Rough real sizes (longest side): water bottle 0.22 m,
  shoe 0.28 m, keys 0.08 m (keys vary most; report as low confidence). Use `max(box_w, box_h)`
  so the estimate does not depend on whether the object is lying or standing.
- Why here: monocular metric depth is unreliable on new scenes (scale is not identifiable
  from one image; recent papers add per-image calibration for that reason), while this
  geometry is exact up to the object's true size. Two independent estimates that agree
  are much more trustworthy than one; where they disagree, say so in the output.
- Also: report the near/far ORDER accuracy separately from metres (order is reliable), and
  fit any scale correction for the metric model with **leave-one-out** on Alex's
  `distance/` photos (never fit and score on the same photo, since those photos are test-only).
- Licence note: Depth Anything V2 Small is Apache-2.0. The metric-indoor Small weights are
  fine-tuned on Hypersim; the weights are labelled Apache-2.0 but check the Hypersim data
  licence before any commercial claim. The size-prior estimate has no such concern.
- Expected gain: likely better metres for bottles and shoes; keys will stay rough.

### Considered but not recommended
- **RF-DETR Nano/Small** (Apache-2.0 code and COCO weights; XL/2XL and Atto-Pico are PML):
  DINOv2 backbone is said to adapt well to small data, but the evidence is vendor-published
  COCO numbers, and swapping the detector is Alex's call. See "For Alex to decide".
- **Depth Anything V3:** the metric Large variant is Apache-2.0 but large; some other DA3
  variants are reported CC-BY-NC (unverified). V2 Small is the right size for free Colab.
- **SAM 3 for auto-labelling:** impressive text-prompted masks, but weights are gated under
  Meta's own licence (terms not confirmed) and need more GPU memory; OWLv2 (Apache-2.0)
  already does the job.
- **Ultralytics YOLO:** easiest tooling, but AGPL-3.0 (copyleft); avoid for a possibly
  commercial project.
- **Parameter-efficient fine-tuning (LoRA etc.):** the detector is only ~19M parameters and
  trains fast; no benefit on a single Colab GPU.

### For Alex to decide
- Do you want to try **RF-DETR Nano** (Apache-2.0) as an alternative detector once we have
  the first real score for the current one? It would be a separate experiment, not a
  replacement, and only worth it if the Faster R-CNN numbers on your own videos are weak.

### Sources
- Ghiasi et al., Simple Copy-Paste, CVPR 2021: https://arxiv.org/abs/2012.07177
- Kisantal et al., Augmentation for small object detection: https://arxiv.org/abs/1902.07296
- RF-DETR repo (licences per model size): https://github.com/roboflow/rf-detr
- RSA, resolving scale ambiguity in monocular depth: https://arxiv.org/abs/2410.02924
- Per-image affine calibration of relative depth (2026 preprint): https://arxiv.org/abs/2601.01457
- Depth Anything 3 metric Large model card: https://huggingface.co/APaul1/DA3METRIC-LARGE
- LightlyTrain depth docs (Apache-2.0 DAv2 small models): https://docs.lightly.ai/train/stable/depth_estimation.html
- SAM 3 overview: https://blog.roboflow.com/what-is-sam3/
