import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "notebooks" / "train_colab.ipynb"


def test_colab_notebook_is_valid_and_uses_real_commands():
    nb = json.loads(NB.read_text())
    assert nb["nbformat"] == 4
    code = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    for cmd in ("claudeknows.train", "claudeknows.evaluate", "claudeknows.predict"):
        assert cmd in code


def test_objects_notebook_uses_real_commands_and_no_personal_data():
    nb = json.loads((NB.parent / "objects_colab.ipynb").read_text())
    assert nb["nbformat"] == 4
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    for cmd in ("claudeknows.openimages", "claudeknows.objects", "claudeknows.train_objects",
                "claudeknows.predict", "evaluate_objects"):
        assert cmd in code
    lines = [l[:len(l) - len(l.lstrip())] + "pass" if l.lstrip().startswith(("!", "%")) else l
             for l in code.splitlines()]
    compile("\n".join(lines), "notebook", "exec")  # the Python cells at least parse
    assert "git add" not in code and "git push" not in code  # footage never leaves Colab
