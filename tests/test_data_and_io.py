import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import prepare_data  # noqa: E402

from netra.data import class_weights, load_hospital, read_labels  # noqa: E402
from netra.io import atomic_write_json, read_json  # noqa: E402


def make_zip(tmp_path, hospital_id, n=40, label_ext=".csv", extra_col=False, name_suffix=""):
    src = tmp_path / "src"
    df = prepare_data.synthetic_frame(src, n, seed=1)
    rows = [(p.name.replace(".png", name_suffix), lvl) for p, lvl in zip(df["path"], df["level"])]
    labels = pd.DataFrame(rows, columns=["image_name", "level"])
    if extra_col:
        labels["note"] = "x"
    zdir = tmp_path / "zips"
    zdir.mkdir(exist_ok=True)
    with zipfile.ZipFile(zdir / f"hospital_{hospital_id}_data.zip", "w") as z:
        for p in df["path"]:
            z.write(p, f"images/{Path(p).name}")
        if label_ext == ".csv":
            z.writestr("labels.csv", labels.to_csv(index=False))
        else:
            buf = tmp_path / "labels.xlsx"
            labels.to_excel(buf, index=False)
            z.write(buf, "labels.xlsx")
    return zdir, tmp_path / "unz"


def test_load_hospital_splits_and_normalises(tmp_path):
    zdir, udir = make_zip(tmp_path, 1)
    x_tr, y_tr, x_te, y_te = load_hospital(1, zdir, udir)
    assert len(x_tr) + len(x_te) == 40
    assert x_tr.shape[1:] == (224, 224, 3)
    assert 0.0 <= x_tr.min() and x_tr.max() <= 1.0
    assert set(y_tr) <= set(range(5))


def test_labels_with_extra_columns_and_extensions(tmp_path):
    zdir, udir = make_zip(tmp_path, 2, extra_col=True, name_suffix=".png")
    x_tr, _, x_te, _ = load_hospital(2, zdir, udir)
    assert len(x_tr) + len(x_te) == 40


def test_excel_labels(tmp_path):
    zdir, udir = make_zip(tmp_path, 3, label_ext=".xlsx")
    x_tr, _, x_te, _ = load_hospital(3, zdir, udir)
    assert len(x_tr) + len(x_te) == 40


def test_out_of_range_grades_dropped(tmp_path):
    (tmp_path / "labels.csv").write_text("image_name,level\na,0\nb,9\nc,2\n")
    df = read_labels(tmp_path)
    assert list(df["image_name"]) == ["a", "c"]


def test_missing_zip_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="prepare_data"):
        load_hospital(7, tmp_path, tmp_path / "u")


def test_class_weights_upweight_rare_grades():
    w = class_weights(np.array([0] * 90 + [4] * 10))
    assert w[4] > w[0]
    assert w[2] == 1.0  # absent class


def test_atomic_json_roundtrip_leaves_no_temp_files(tmp_path):
    p = tmp_path / "log.json"
    atomic_write_json(p, {"a": 1})
    atomic_write_json(p, {"a": 2})
    assert read_json(p) == {"a": 2}
    assert [f.name for f in tmp_path.iterdir()] == ["log.json"]


def test_read_json_default_when_missing_or_corrupt(tmp_path):
    assert read_json(tmp_path / "nope.json", {"x": 0}) == {"x": 0}
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert read_json(bad, "d") == "d"


def test_skewed_split_changes_class_mix():
    df = pd.DataFrame({"path": range(2000), "level": np.random.default_rng(0).integers(0, 5, 2000)})
    parts = prepare_data.split_for_hospitals(df, 3, 300, skew=1.0, seed=0)
    mixes = [np.bincount(p["level"], minlength=5) / 300 for p in parts]
    assert max(np.abs(mixes[0] - mixes[1])) > 0.1
