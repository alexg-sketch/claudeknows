# Progress log

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
- Builds on `claude/2026-09-29`, which already contains the work from PRs #2-#4
  (those were merged into that branch, not into `main`). Merge this PR into
  `claude/2026-09-29`, then that branch into `main`.

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
