# claudeknows roadmap

Image classifier + decision maker: look at an image, predict a label, then pick
an action from that label (and how confident the model is).

## Notes from Alex
<!-- Add notes here. Claude reads this section first on every run. -->
- 2026-09-30: Focus is only on getting the model to understand what it's looking at
  (accurate classification). The decision layer is parked until Alex says otherwise.
- 2026-10-07: Alex trained the model on Colab: 87.8% test accuracy. Keep the trained
  model in `models/cifar10_smallcnn.pt`. Every pull request must target `main`.
- 2026-10-07: Next goal: teach it new objects from Alex's own photos. Alex will put
  photos in Google Drive, one folder per object (e.g. `claudeknows_photos/mug/`).
  The repo is PUBLIC: never commit Alex's photos to it. Training happens in Colab.
  Start from a small pretrained network (transfer learning) so ~30 photos per
  object is enough. The current CIFAR-10 model and web page must keep working.

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
- [ ] 14. Own-photos dataset: load a folder with one sub-folder per object (any number of objects), split into train/test, resize for a pretrained network, light augmentation; works with a tiny generated folder of fake images for tests
- [ ] 15. Transfer-learning model + training: small pretrained network (e.g. MobileNetV3-small or ResNet18 from torchvision) with a new final layer sized to the number of folders; save the class names inside the checkpoint; `predict` and `evaluate` work with it
- [ ] 16. Colab notebook for own photos: connect Google Drive, point at the photos folder, check how many photos each object has (warn under 20), train, show accuracy per object and the photos it got wrong, download the model
- [ ] 17. Web page for the own-photos model (choose how to run the bigger network in the browser)
