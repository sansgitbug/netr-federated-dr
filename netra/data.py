"""Hospital-side data loading: unzip, read labels, load images, split."""
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from .config import DATA_UNZIP_DIR, DATA_ZIP_DIR, IMAGE_SHAPE, NUM_CLASSES

IMAGE_EXTS = (".jpg", ".jpeg", ".png")
MAX_IMAGES = 5000  # everything is held in RAM; fail loudly instead of swapping


def unzip_data(hospital_id, zip_dir=DATA_ZIP_DIR, out_dir=DATA_UNZIP_DIR) -> Path:
    zip_path = Path(zip_dir) / f"hospital_{hospital_id}_data.zip"
    target = Path(out_dir) / f"hospital_{hospital_id}_data"
    if target.is_dir() and any(target.iterdir()):
        return target
    if not zip_path.exists():
        raise FileNotFoundError(
            f"Data zip not found: {zip_path}. Run scripts/prepare_data.py first."
        )
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(target)
    return target


def _find_root(base: Path) -> Path:
    """Zips may contain the files directly or inside one nested folder."""
    if (base / "images").is_dir():
        return base
    for child in base.iterdir():
        if child.is_dir() and (child / "images").is_dir():
            return child
    raise FileNotFoundError(f"No 'images' folder found under {base}")


def read_labels(root: Path) -> pd.DataFrame:
    for name in ("labels.csv", "labels.xlsx"):
        p = root / name
        if p.exists():
            df = pd.read_csv(p) if name.endswith(".csv") else pd.read_excel(p)
            break
    else:
        raise FileNotFoundError(f"No labels.csv or labels.xlsx in {root}")
    if df.shape[1] < 2:
        raise ValueError("Label file needs at least two columns: image name, grade")
    df = df.iloc[:, :2].copy()
    df.columns = ["image_name", "level"]
    df["level"] = pd.to_numeric(df["level"], errors="coerce")
    bad = df["level"].isna() | ~df["level"].isin(range(NUM_CLASSES))
    if bad.any():
        print(f"  Dropping {int(bad.sum())} label rows with grades outside 0-{NUM_CLASSES - 1}")
        df = df[~bad]
    df["level"] = df["level"].astype(int)
    return df


def _find_image(image_dir: Path, name: str):
    name = str(name).strip()
    stem, ext = os.path.splitext(name)
    candidates = [name] if ext.lower() in IMAGE_EXTS else []
    candidates += [stem + e for e in IMAGE_EXTS] if ext.lower() in IMAGE_EXTS else [name + e for e in IMAGE_EXTS]
    for c in candidates:
        p = image_dir / c
        if p.exists():
            return p
    return None


def load_hospital(hospital_id, zip_dir=DATA_ZIP_DIR, out_dir=DATA_UNZIP_DIR, test_size=0.2, seed=42):
    """Return (x_train, y_train, x_test, y_test) with y as integer grades."""
    root = _find_root(unzip_data(hospital_id, zip_dir, out_dir))
    df = read_labels(root)
    image_dir = root / "images"

    xs, ys, missing = [], [], 0
    for name, level in zip(df["image_name"], df["level"]):
        path = _find_image(image_dir, name)
        if path is None:
            missing += 1
            continue
        try:
            with Image.open(path) as im:
                arr = np.asarray(im.convert("RGB").resize(IMAGE_SHAPE), dtype="float32") / 255.0
        except Exception as e:  # corrupt image
            print(f"  Skipping unreadable image {path.name}: {e}")
            missing += 1
            continue
        xs.append(arr)
        ys.append(level)
        if len(xs) > MAX_IMAGES:
            raise MemoryError(f"More than {MAX_IMAGES} images; use a smaller per-hospital slice.")

    if missing:
        print(f"  {missing} of {len(df)} labelled images were missing or unreadable")
    if len(xs) < 10:
        raise ValueError(f"Only {len(xs)} usable images for hospital {hospital_id}")

    x, y = np.stack(xs), np.array(ys, dtype="int64")
    counts = np.bincount(y, minlength=NUM_CLASSES)
    can_stratify = counts[counts > 0].min() >= 2
    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=test_size, random_state=seed, stratify=y if can_stratify else None
    )
    return x_tr, y_tr, x_te, y_te


def class_weights(y, num_classes=NUM_CLASSES):
    """Balanced weights for the classes present; absent classes get 1.0."""
    counts = np.bincount(y, minlength=num_classes).astype("float64")
    present = counts > 0
    w = np.ones(num_classes)
    w[present] = counts.sum() / (present.sum() * counts[present])
    return {i: float(w[i]) for i in range(num_classes)}
