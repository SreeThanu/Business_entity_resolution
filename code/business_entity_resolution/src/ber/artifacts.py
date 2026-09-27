"""Checkpoint and model-directory helpers for scripts/07-10 (all under config.PERSIST_DIR).

Every stage writes its outputs atomically (temp name, then rename), so a file that exists is
complete, and a re-run after a disconnect skips it. A folder of chunks records the settings it was
made with in run.json; resuming with different settings is refused instead of mixing them.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import polars as pl

from ber import config

CODE_DIR = Path(__file__).resolve().parents[2]


def log(msg: str) -> None:
    from ber.memguard import rss_gib
    print(f"[{time.strftime('%H:%M:%S')} rss {rss_gib():.2f} GiB] {msg}", flush=True)


def atomic_write_parquet(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.write_parquet(tmp, compression="zstd")
    tmp.replace(path)


def atomic_write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    tmp.replace(path)


def sha256_lines(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def check_run_manifest(folder: Path, fingerprint: dict) -> None:
    """Record `fingerprint` in folder/run.json, or refuse if the folder was made with another one."""
    folder.mkdir(parents=True, exist_ok=True)
    manifest = folder / "run.json"
    if manifest.exists():
        old = json.loads(manifest.read_text())
        if old != json.loads(json.dumps(fingerprint)):
            raise RuntimeError(f"{folder} was made with different settings; re-run with --force to start over.\n"
                               f"then: {json.dumps(old)}\nnow:  {json.dumps(fingerprint, default=str)}")
    else:
        atomic_write_json(fingerprint, manifest)


def reset_folder(folder: Path) -> None:
    if folder.exists():
        shutil.rmtree(folder)


def git_commit() -> str:
    """Short commit hash of the code, '-dirty' if tracked files have uncommitted changes."""
    try:
        run = lambda *a: subprocess.run(["git", "-C", str(CODE_DIR), *a], capture_output=True, text=True, check=True).stdout.strip()
        h = run("rev-parse", "--short=10", "HEAD")
        return h + ("-dirty" if run("status", "--porcelain", "--untracked-files=no") else "")
    except (OSError, subprocess.CalledProcessError):
        return "nogit"


# ---------------------------------------------------------------------------------------------
# Feature folders
# ---------------------------------------------------------------------------------------------

def features_dir(split: str) -> Path:
    return config.FEATURES_DIR / split


def feature_parts(split: str) -> list[Path]:
    """The complete chunk files of a feature split; raises if 07_features.py has not finished it."""
    d = features_dir(split)
    done = d / "_DONE.json"
    if not done.exists():
        raise FileNotFoundError(f"features for {split!r} are not complete ({done} missing): "
                                f"run scripts/07_features.py --split {split}")
    parts = json.loads(done.read_text())["parts"]
    return [d / p for p in parts]


def feature_fingerprint(split: str) -> dict:
    return json.loads((features_dir(split) / "_DONE.json").read_text())["fingerprint"]


# ---------------------------------------------------------------------------------------------
# Model folders: MODELS_DIR/<YYYYmmdd-HHMMSS>_<commit>/
# ---------------------------------------------------------------------------------------------

def new_model_id() -> str:
    return f"{time.strftime('%Y%m%d-%H%M%S')}_{git_commit()}"


def model_dirs() -> list[Path]:
    if not config.MODELS_DIR.exists():
        return []
    return sorted(p for p in config.MODELS_DIR.iterdir() if p.is_dir() and (p / "model.txt").exists())


def resolve_model(model_id: str | None = None, need: str | None = None) -> Path:
    """The model folder named `model_id` (or $BER_MODEL_ID), else the newest one that contains `need`."""
    model_id = model_id or os.environ.get("BER_MODEL_ID")
    if model_id:
        d = config.MODELS_DIR / model_id
        if not (d / "model.txt").exists():
            raise FileNotFoundError(f"no model at {d}")
        return d
    cands = [d for d in model_dirs() if need is None or (d / need).exists()]
    if not cands:
        what = f"a model with {need}" if need else "a model"
        raise FileNotFoundError(f"no {what} in {config.MODELS_DIR}; run 08_train.py" + (" and 09_evaluate.py" if need else ""))
    return cands[-1]
