import json
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

from claudeknows.export_web import DEFAULT_CHECKPOINT, build_page, pack_weights
from claudeknows.model import SmallCNN

ROOT = Path(__file__).resolve().parent.parent


def test_build_page_fills_placeholders():
    full, fragment = build_page(DEFAULT_CHECKPOINT)
    for page in (full, fragment):
        assert "__MODEL_" not in page
        assert "/*__CLASSIFIER_JS__*/" not in page
        assert "<title>claudeknows</title>" in page
    assert full.startswith("<!doctype html>")
    assert "<body>" not in fragment


def test_committed_page_is_up_to_date():
    full, _ = build_page(DEFAULT_CHECKPOINT)
    assert (ROOT / "web" / "index.html").read_text() == full, (
        "web/index.html is stale: run python -m claudeknows.export_web"
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
@pytest.mark.parametrize("trained", [True, False])
def test_javascript_matches_pytorch(tmp_path, trained):
    torch.manual_seed(0)
    model = SmallCNN()
    if trained:
        state = torch.load(DEFAULT_CHECKPOINT, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model_state"])
    else:
        # Randomise batch-norm stats so folding is actually exercised.
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.running_mean.uniform_(-0.5, 0.5)
                m.running_var.uniform_(0.5, 2.0)
                m.weight.data.uniform_(0.5, 1.5)
                m.bias.data.uniform_(-0.2, 0.2)
    model.eval()
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        expected = model(x)[0]

    meta, flat = pack_weights(model)
    (tmp_path / "weights.bin").write_bytes(flat.tobytes())
    (tmp_path / "input.bin").write_bytes(x.numpy().astype("<f4").tobytes())
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    script = f"""
    const fs = require("fs");
    const c = require({json.dumps(str(ROOT / "web" / "classifier.js"))});
    const dir = {json.dumps(str(tmp_path))};
    const meta = JSON.parse(fs.readFileSync(dir + "/meta.json"));
    const asF32 = (b) => new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.length));
    const model = c.buildModel(meta, asF32(fs.readFileSync(dir + "/weights.bin")));
    const out = c.forward(model, asF32(fs.readFileSync(dir + "/input.bin")));
    console.log(JSON.stringify(Array.from(out)));
    """
    result = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True)
    got = torch.tensor(json.loads(result.stdout))
    assert torch.allclose(got, expected, atol=1e-3), (got, expected)
