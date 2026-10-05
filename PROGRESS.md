# Progress log

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
