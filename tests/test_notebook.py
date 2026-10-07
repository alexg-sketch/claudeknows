import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "notebooks" / "train_colab.ipynb"


def test_colab_notebook_is_valid_and_uses_real_commands():
    nb = json.loads(NB.read_text())
    assert nb["nbformat"] == 4
    code = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    for cmd in ("claudeknows.train", "claudeknows.evaluate", "claudeknows.predict"):
        assert cmd in code
