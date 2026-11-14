"""Atomic file writes so the dashboard never reads a half-written file."""
import json
import os
import tempfile
import time
from pathlib import Path


def replace_with_retry(src, dst, attempts: int = 40, delay: float = 0.05) -> None:
    """os.replace that tolerates Windows briefly locking `dst` while another process reads it."""
    for i in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(delay)


def atomic_write_bytes(path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        replace_with_retry(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def atomic_write_json(path, obj) -> None:
    atomic_write_bytes(path, json.dumps(obj, indent=2).encode("utf-8"))


def read_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return default


def save_weights_atomic(model, path) -> None:
    """Save Keras weights via a temp file, then swap into place."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("tmp_" + path.name)  # Keras 3 needs the name to end in .weights.h5
    model.save_weights(str(tmp))
    replace_with_retry(tmp, path)
