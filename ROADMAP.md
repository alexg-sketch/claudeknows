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
- [ ] 16. Colab notebook for the new objects: download the photos (task 14), show a few per object, train, show accuracy per object and the photos it got wrong, download the model
- [ ] 17. *(parked: replaced by task 23, the web page for the spatial model)* Web page for the classifier
- [ ] 18. Detection data: from Open Images, keep whole photos with their boxes (bottle boxes in water-bottle photos, footwear boxes), several boxes per photo allowed; save as images + one JSON of boxes per object folder; credits file; plus `together/` and `distance/` folders from Alex's Drive (see notes). Tests with a tiny generated dataset
- [ ] 19. Auto-boxing with OWLv2 (Colab only): boxes for key photos and for every frame of Alex's footage, confidence threshold, contact sheet of boxes for a quick human look; mocked in tests
- [ ] 20. Detector model + training: pretrained torchvision detector with a 3-object head; box-aware augmentation for "any angle"; evaluation: mAP@0.5 per object and per source, plus breakdown by box size (near/far), position (centre/edge) and held-out videos; tests with `weights=None`
- [ ] 21. Distance: Depth Anything V2 Small (relative) + metric-indoor Small (rough metres) per detected box; near->far ordering; check against Alex's `distance/` photos and report the error in plain words
- [ ] 22. Colab notebook for the spatial model: download + auto-box + train + evaluate + distance check; upload a photo and see boxes, labels, confidence and distances drawn on it; download the model
- [ ] 23. Web page for the spatial model: show boxes and distances on an uploaded photo (decide how to run it in the browser, or link to the notebook if it's too big)
