import torch

from claudeknows.data import IMAGE_SHAPE, NUM_CLASSES
from claudeknows.model import SmallCNN, count_parameters
from claudeknows.train import train


def test_model_output_shape():
    out = SmallCNN()(torch.randn(4, *IMAGE_SHAPE))
    assert out.shape == (4, NUM_CLASSES)


def test_param_count_in_range():
    assert 100_000 <= count_parameters(SmallCNN()) <= 1_000_000


def test_smoke_train_saves_checkpoint(tmp_path):
    ckpt = tmp_path / "ck.pt"
    hist = train(epochs=1, batch_size=16, fake=True, max_batches=2,
                 checkpoint=str(ckpt), num_workers=0, root=str(tmp_path))
    assert len(hist) == 1 and ckpt.exists()
    state = torch.load(ckpt)["model_state"]
    m = SmallCNN()
    m.load_state_dict(state)
