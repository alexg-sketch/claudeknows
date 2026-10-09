# claudeknows

An image classifier built with PyTorch: it looks at an image and predicts what
it is (one of the 10 CIFAR-10 classes: airplane, automobile, bird, cat, deer,
dog, frog, horse, ship, truck), along with how confident it is.

> The "decision layer" (turning a label + confidence into an action) is parked
> for now. The current focus is accurate classification.

## Install

```bash
pip install -r requirements.txt   # torch, torchvision, pytest, opencv (video frames)
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
python -m claudeknows.predict path/to/image.jpg --checkpoint models/cifar10_smallcnn.pt
```

Prints the predicted label and confidence (plus the top few guesses). Images
are resized to 32x32 first, so small, simple pictures work best.

## Try it in your browser

Open `web/index.html` in any browser (double-click it), choose a photo, and it
shows what the model thinks it is, how sure it is, and the 32x32 version of
the photo the model actually looks at. Everything runs in the browser; the
photo is never uploaded anywhere.

The page uses the trained model in `models/cifar10_smallcnn.pt` (87.8% test
accuracy after 20 epochs on Colab). After training a new model, rebuild the
page:

```bash
python -m claudeknows.export_web --checkpoint models/cifar10_smallcnn.pt
```

## New objects: water bottle, keys, shoes

A second model learns these three from photos, starting from a small
pretrained network (MobileNetV3-small). Training photos come from two places
and are combined:

1. **Internet photos** (Google's Open Images, CC BY):
   `python -m claudeknows.openimages --out data/objects --cap 300`
   (photos are cropped to the object; `data/objects/web/CREDITS.md` lists the
   photographers).
2. **Your own phone videos/photos**, optional, put in a folder with
   `water_bottle/`, `keys/` and `shoes/` sub-folders (e.g. Google Drive's
   `claudeknows_photos/`):
   `python -m claudeknows.objects /path/to/claudeknows_photos`
   Videos (including iPhone `.MOV`) become ~2 frames per second, blurry frames
   are skipped. A missing or empty folder is fine.

Then:

```bash
python -m claudeknows.train_objects --root data/objects --epochs 8 --checkpoint objects.pt
python -m claudeknows.train_objects --evaluate --checkpoint objects.pt
python -m claudeknows.predict photo.jpg --checkpoint objects.pt
```

The test photos are held out **by video** (all frames of a video are either
training or test, never both), and the report shows accuracy per object and
per source (internet vs your own footage), plus the photos it got wrong.

**Privacy:** the repo is public. `claudeknows_photos/`, `data/` and video/HEIC
files are git-ignored; never force-add them.

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
- `models/cifar10_smallcnn.pt` - the trained model (from the Colab run)
- `claudeknows/export_web.py`, `web/` - the in-browser photo page (`web/classifier.js` runs the network in plain JavaScript)
- `claudeknows/openimages.py`, `frames.py`, `objects.py` - get internet photos, turn videos into frames, combine and split the data
- `claudeknows/transfer.py`, `train_objects.py` - the pretrained-network model for the new objects
