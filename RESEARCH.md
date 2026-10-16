# Research log

Daily notes from the research advisor. The dev routine reads the roadmap tasks tagged
`[research YYYY-MM-DD]`; this file holds the reasoning and sources behind them.

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
