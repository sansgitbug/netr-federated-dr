"""Create per-hospital data zips (images/ + labels.csv) for the federated demo.

Real data (APTOS 2019 style: a CSV with id_code + diagnosis columns and a folder of images):
    python scripts/prepare_data.py --source <folder> --hospitals 3 --per-hospital 300
    python scripts/prepare_data.py --download --hospitals 3     # via kagglehub; needs Kaggle login
Placeholder data for smoke tests (not medical images):
    python scripts/prepare_data.py --synthetic --hospitals 3 --per-hospital 60

--skew 0 gives every hospital a similar grade mix; larger values (e.g. 1.0) give each
hospital a different mix, to mimic real non-IID data.
"""
import argparse
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from netra.config import DATA_ZIP_DIR, NUM_CLASSES  # noqa: E402

IMG_EXTS = {".png", ".jpg", ".jpeg"}


def find_labelled_images(source: Path) -> pd.DataFrame:
    """Return DataFrame[path, level] from an APTOS-style folder."""
    # Kaggle folders also hold sample_submission.csv (same columns, all-zero dummy labels),
    # so choose the candidate with the most distinct grades, not just the first match.
    label_df, best = None, 0
    for csv in sorted(source.rglob("*.csv")):
        try:
            df = pd.read_csv(csv)
        except Exception:
            continue
        if {"id_code", "diagnosis"} <= set(df.columns):
            distinct = df["diagnosis"].nunique()
            if distinct > best:
                label_df, best = df[["id_code", "diagnosis"]], distinct
                print(f"Using labels from {csv.name} ({distinct} distinct grades)")
    if label_df is None:
        raise SystemExit(f"No CSV with 'id_code' and 'diagnosis' columns found under {source}")
    index = {}
    for p in source.rglob("*"):
        if p.suffix.lower() in IMG_EXTS:
            index.setdefault(p.stem, p)
    rows = [(index[str(i)], int(d)) for i, d in zip(label_df["id_code"], label_df["diagnosis"])
            if str(i) in index]
    if not rows:
        raise SystemExit("Found the label CSV but no matching images.")
    print(f"Found {len(rows)} labelled images ({len(label_df) - len(rows)} labels without an image).")
    return pd.DataFrame(rows, columns=["path", "level"])


def synthetic_frame(out_dir: Path, n: int, seed: int) -> pd.DataFrame:
    """Learnable placeholder images: a disc whose number of bright 'lesions' grows with grade."""
    from PIL import Image, ImageDraw

    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    levels = rng.choice(NUM_CLASSES, size=n, p=[0.45, 0.15, 0.25, 0.08, 0.07])
    for i, lvl in enumerate(levels):
        img = Image.new("RGB", (96, 96), (10, 10, 10))
        d = ImageDraw.Draw(img)
        d.ellipse((6, 6, 90, 90), fill=(150 + int(rng.integers(0, 40)), 70, 30))
        for _ in range(int(lvl) * 4):
            x, y = rng.integers(14, 82, size=2)
            d.ellipse((x, y, x + 4, y + 4), fill=(250, 240, 120))
        p = out_dir / f"syn_{i:05d}.png"
        img.save(p)
        rows.append((p, int(lvl)))
    return pd.DataFrame(rows, columns=["path", "level"])


def split_for_hospitals(df, hospitals, per_hospital, skew, seed):
    rng = np.random.default_rng(seed)
    pool = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    parts = []
    for _ in range(hospitals):
        if len(pool) < per_hospital:
            raise SystemExit(f"Not enough images: need {per_hospital} per hospital, "
                             f"{len(pool)} left.")
        if skew > 0:
            class_w = rng.dirichlet(np.full(NUM_CLASSES, 1.0 / skew))
            p = class_w[pool["level"].to_numpy()]
            p = p / p.sum()
            idx = rng.choice(len(pool), size=per_hospital, replace=False, p=p)
        else:
            idx = np.arange(per_hospital)
        parts.append(pool.iloc[idx])
        pool = pool.drop(pool.index[idx]).reset_index(drop=True)
    return parts


def write_zip(hospital_id: int, part: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    zpath = out_dir / f"hospital_{hospital_id}_data.zip"
    labels = []
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for path, level in zip(part["path"], part["level"]):
            path = Path(path)
            z.write(path, f"images/{path.name}")
            labels.append((path.name, int(level)))
        z.writestr("labels.csv", pd.DataFrame(labels, columns=["image_name", "level"]).to_csv(index=False))
    counts = np.bincount(part["level"], minlength=NUM_CLASSES).tolist()
    print(f"  hospital {hospital_id}: {len(part)} images, grade counts {counts} -> {zpath}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--source", type=Path, help="folder with the labels CSV and images")
    src.add_argument("--download", action="store_true", help="download APTOS 2019 with kagglehub")
    src.add_argument("--synthetic", action="store_true", help="generate placeholder images")
    ap.add_argument("--hospitals", type=int, default=3)
    ap.add_argument("--per-hospital", type=int, default=300)
    ap.add_argument("--skew", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=DATA_ZIP_DIR)
    args = ap.parse_args()

    if args.synthetic:
        tmp = args.out.parent / "data_synthetic_src"
        df = synthetic_frame(tmp, args.hospitals * args.per_hospital, args.seed)
    else:
        source = args.source
        if args.download:
            import kagglehub

            print("Downloading APTOS 2019 (accept the competition rules on Kaggle first)...")
            source = Path(kagglehub.competition_download("aptos2019-blindness-detection"))
        df = find_labelled_images(source)

    print(f"Writing {args.hospitals} hospital zips (skew={args.skew}):")
    for i, part in enumerate(split_for_hospitals(df, args.hospitals, args.per_hospital,
                                                 args.skew, args.seed), start=1):
        write_zip(i, part, args.out)


if __name__ == "__main__":
    main()
