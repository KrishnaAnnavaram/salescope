"""Save and load a fitted pipeline with a metadata file next to it."""

from __future__ import annotations

import hashlib
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn

MODEL_FILE = "model.joblib"
META_FILE = "metadata.json"


def frame_fingerprint(df: pd.DataFrame) -> str:
    """SHA-256 of the frame content, so a saved model records exactly which rows trained it."""
    hashed = pd.util.hash_pandas_object(df, index=False).to_numpy()
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def save_model(model, out_dir: str | Path, metadata: dict) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / MODEL_FILE)
    meta = {
        **metadata,
        "sklearn_version": sklearn.__version__,
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out / META_FILE).write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return out


def load_model(model_dir: str | Path):
    """Return ``(model, metadata)``. Warns when the scikit-learn version differs from the saved one."""
    d = Path(model_dir)
    if not (d / MODEL_FILE).is_file():
        raise FileNotFoundError(f"{d / MODEL_FILE} does not exist. Run `salescope train` first.")
    meta = json.loads((d / META_FILE).read_text(encoding="utf-8")) if (d / META_FILE).is_file() else {}
    saved = meta.get("sklearn_version")
    if saved and saved != sklearn.__version__:
        warnings.warn(f"model saved with scikit-learn {saved}, running {sklearn.__version__}", stacklevel=2)
    return joblib.load(d / MODEL_FILE), meta
