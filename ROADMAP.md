# claudeknows roadmap

Image classifier + decision maker: look at an image, predict a label, then pick
an action from that label (and how confident the model is).

## Notes from Alex
<!-- Add notes here. Claude reads this section first on every run. -->

## Tasks
- [x] 1. Project setup: package layout, `requirements.txt`, pytest config, `.gitignore`
- [x] 2. Dataset loader: CIFAR-10 train/test `DataLoader`s with normalisation + augmentation, plus a synthetic "fake" mode for offline tests
- [ ] 3. Small CNN in PyTorch (a few conv blocks, ~100k–1M params) with a shape test
- [ ] 4. Training script (`python -m claudeknows.train`): epochs, lr, batch size, device auto-detect, `--max-batches` for smoke runs, saves a checkpoint
- [ ] 5. Evaluation script: overall accuracy, per-class accuracy, confusion matrix
- [ ] 6. Prediction helper: load checkpoint, run on a single image file, return label + confidence
- [ ] 7. Decision layer: map (label, confidence) to an action via a config (with a "not sure" fallback below a confidence threshold)
- [ ] 8. End-to-end CLI: `image -> label -> action`
- [ ] 9. Tests for model, training step, evaluation, and decision layer (all run on CPU in seconds)
- [ ] 10. Colab notebook for full GPU training that clones the repo, trains, evaluates, and saves the checkpoint
- [ ] 11. README: how to install, train, run predictions, and customise actions
