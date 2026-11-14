"""Single source of truth for names, paths and run settings."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

NUM_CLASSES = 5
IMAGE_SHAPE = (224, 224)
CLASS_NAMES = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR",
]
# Grades at or above this are "referable" (should see an ophthalmologist).
REFERABLE_FROM = 2

HOSPITAL_NAMES = {
    1: "City General Hospital",
    2: "Suburban Medical Center",
    3: "Rural District Hospital",
    4: "University Medical Research",
}


def hospital_name(hospital_id) -> str:
    try:
        return HOSPITAL_NAMES.get(int(hospital_id), f"Hospital {hospital_id}")
    except (TypeError, ValueError):
        return f"Hospital {hospital_id}"


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


MODEL_DIR = Path(_env("NETRA_MODEL_DIR", str(ROOT / "models")))
MODEL_PATH = MODEL_DIR / "global_model.weights.h5"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
LOGS_PATH = Path(_env("NETRA_LOGS_PATH", str(ROOT / "logs" / "training_metrics.json")))
DATA_ZIP_DIR = Path(_env("NETRA_DATA_ZIPS", str(ROOT / "data_zips")))
DATA_UNZIP_DIR = Path(_env("NETRA_DATA_UNZIPPED", str(ROOT / "data_unzipped")))
REPLAY_LOG_PATH = ROOT / "demo" / "training_metrics.json"

SERVER_HOST = _env("NETRA_SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(_env("NETRA_SERVER_PORT", "8080"))
NUM_ROUNDS = int(_env("NETRA_ROUNDS", "10"))
CLIENTS_REQUIRED = int(_env("NETRA_CLIENTS", "3"))
LOCAL_EPOCHS = int(_env("NETRA_LOCAL_EPOCHS", "3"))
LEARNING_RATE = float(_env("NETRA_LR", "1e-4"))
# Class-balanced loss was unstable on small hospital slices (solo test: kappa swung 0.31-0.88).
USE_CLASS_WEIGHTS = _env("NETRA_CLASS_WEIGHTS", "0") == "1"
# Keep BatchNorm layers in inference mode: averaging their running stats across
# very different hospitals is a known cause of federated collapse.
FREEZE_BATCHNORM = _env("NETRA_FREEZE_BN", "1") == "1"
BATCH_SIZE = int(_env("NETRA_BATCH_SIZE", "16"))
