# Research log

Daily notes from the research advisor. The dev routine reads the roadmap tasks tagged
`[research YYYY-MM-DD]`; this file holds the reasoning and sources behind them.

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
