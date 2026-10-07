# claudeknows

An image classifier built with PyTorch: it looks at an image and predicts what
it is (one of the 10 CIFAR-10 classes: airplane, automobile, bird, cat, deer,
dog, frog, horse, ship, truck), along with how confident it is.

> The "decision layer" (turning a label + confidence into an action) is parked
> for now. The current focus is accurate classification.

## Install

```bash
pip install -r requirements.txt   # torch, torchvision, pytest
```

## Train

```bash
python -m claudeknows.train --epochs 20 --checkpoint checkpoint.pt
```

Options: `--epochs`, `--lr`, `--batch-size`, `--root` (where CIFAR-10 is
stored), `--checkpoint`, `--num-workers`. The GPU is used automatically if
there is one.

Quick smoke run with no download (synthetic images, a couple of batches):

```bash
python -m claudeknows.train --fake --epochs 1 --max-batches 2
```

For real GPU training, open `notebooks/train_colab.ipynb` in Google Colab. It
clones the repo, trains, evaluates, and saves the checkpoint.

## Evaluate

```bash
python -m claudeknows.evaluate --checkpoint checkpoint.pt
```

Prints overall accuracy, per-class accuracy, and a confusion matrix. `--fake`
and `--max-batches` work here too for quick checks.

## Predict on one image

```bash
python -m claudeknows.predict path/to/image.jpg --checkpoint checkpoint.pt
```

Prints the predicted label and confidence (plus the top few guesses). Images
are resized to 32x32 first, so small, simple pictures work best.

## Tests

```bash
pytest
```

All tests run on CPU in seconds and need no dataset download.

## Layout

- `claudeknows/data.py` - CIFAR-10 loaders (with a `fake=True` offline mode)
- `claudeknows/model.py` - `SmallCNN`
- `claudeknows/train.py`, `evaluate.py`, `predict.py` - the command-line tools
- `notebooks/train_colab.ipynb` - GPU training on Colab
