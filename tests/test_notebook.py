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


def test_spatial_notebook_uses_real_commands_and_no_personal_data():
    nb = json.loads((NB.parent / "spatial_colab.ipynb").read_text())
    assert nb["nbformat"] == 4
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    for cmd in ("claudeknows.detection_data download", "claudeknows.detection_data import",
                "claudeknows.autobox run", "claudeknows.detector train",
                "claudeknows.detector report", "claudeknows.distance check", "draw_results"):
        assert cmd in code
    lines = [l[:len(l) - len(l.lstrip())] + "pass" if l.lstrip().startswith(("!", "%")) else l
             for l in code.splitlines()]
    compile("\n".join(lines), "notebook", "exec")
    assert "git add" not in code and "git push" not in code


def test_spatial_notebook_saves_model_to_drive():
    nb = json.loads((NB.parent / "spatial_colab.ipynb").read_text())
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert "claudeknows_models" in code and "detector.pt" in code


def test_try_notebook_loads_saved_model_without_training():
    nb = json.loads((NB.parent / "try_spatial.ipynb").read_text())
    assert nb["nbformat"] == 4
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    for name in ("load_detector", "detect", "depth_models", "estimate_distances", "draw_results",
                 "claudeknows_models/detector.pt"):
        assert name in code
    assert "detector train" not in code  # quick: no training here
    lines = [l[:len(l) - len(l.lstrip())] + "pass" if l.lstrip().startswith(("!", "%")) else l
             for l in code.splitlines()]
    compile("\n".join(lines), "notebook", "exec")
    assert "git add" not in code and "git push" not in code


def test_notebooks_open_iphone_photos_and_save_before_trying():
    for name in ("spatial_colab.ipynb", "try_spatial.ipynb"):
        nb = json.loads((NB.parent / name).read_text())
        code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
        assert "open_photo(" in code and "Image.open(name)" not in code
    nb = json.loads((NB.parent / "spatial_colab.ipynb").read_text())
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    save = next(i for i, s in enumerate(code) if "claudeknows_models" in s)
    upload = next(i for i, s in enumerate(code) if "open_photo(" in s)
    assert save < upload  # a bad photo can't stop the model being saved


def test_learning_loop_notebook_runs_loop_and_asks_questions_last():
    nb = json.loads((NB.parent / "learning_loop.ipynb").read_text())
    assert nb["nbformat"] == 4
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    code = "\n".join(cells)
    for name in ("loop.run_loop", "colab_label_fn", "colab_retrain_fn", "claudeknows_models",
                 "claudeknows_photos", "open_questions", "loop.answer", "on_click"):
        assert name in code
    ask = cells[-1]  # the questions come last: nothing after them depends on clicks
    assert "question_widget" in ask and "run_loop" not in ask
    assert "score" not in ask and "why" not in ask  # the view does not nudge the answer
    lines = [l[:len(l) - len(l.lstrip())] + "pass" if l.lstrip().startswith(("!", "%")) else l
             for l in code.splitlines()]
    compile("\n".join(lines), "notebook", "exec")
    assert "git add" not in code and "git push" not in code
