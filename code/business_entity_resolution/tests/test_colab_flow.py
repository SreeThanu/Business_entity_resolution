"""Tests for the Colab plumbing: data manifest, fetch/verify step, config paths."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path

import pytest

CODE = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, CODE / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_manifest_lists_the_pipeline_inputs_only():
    m = json.loads((CODE / "colab" / "data_manifest.json").read_text())
    paths = {e["path"] for e in m["entries"]}
    for need in ("cand/cand_train_200k_candidates.parquet", "cand/cand_val_50k_candidates.parquet",
                 "cand/test_candidates.parquet", "parquet/train_ground_truth_long.parquet",
                 "splits/cand_train_200k.txt", "splits/cand_val_50k.txt", "dicts/state_map.parquet",
                 *(f"clean/{p}{split}_source{s}.parquet" for p in ("", "stage2_") for split in ("train", "test")
                   for s in (1, 2, 3))):
        assert need in paths, need
    assert not any(p.startswith(("cand/cache", "cand/dev")) for p in paths)  # no blocking caches
    assert m["total_bytes"] == sum(e["bytes"] for e in m["entries"]) and m["files"] == len(paths)
    assert all(re.fullmatch(r"[0-9a-f]{64}", e["sha256"]) for e in m["entries"])


def test_fetch_copies_verifies_and_skips(tmp_path, monkeypatch):
    fetch = _load_script("fetch_data")
    src, dst = tmp_path / "drive", tmp_path / "local"
    (src / "clean").mkdir(parents=True)
    (src / "clean" / "a.parquet").write_bytes(b"hello")
    entry = {"path": "clean/a.parquet", "bytes": 5, "sha256": hashlib.sha256(b"hello").hexdigest()}
    assert fetch.fetch(entry, src, dst) is None and (dst / "clean" / "a.parquet").read_bytes() == b"hello"
    (src / "clean" / "a.parquet").unlink()               # already verified locally: no copy needed
    assert fetch.fetch(entry, src, dst) is None
    (src / "clean" / "b.parquet").write_bytes(b"hellx")  # same size, wrong content
    bad = {**entry, "path": "clean/b.parquet"}
    assert "sha256 mismatch" in fetch.fetch(bad, src, dst)
    assert not (dst / "clean" / "b.parquet").exists()
    assert "not found" in fetch.fetch({**entry, "path": "clean/c.parquet"}, src, dst)


def test_no_hard_coded_machine_paths():
    root = CODE.parents[1]
    offenders = []
    for p in list(CODE.rglob("*")) + list((root / "eda").rglob("*")):
        if p.is_file() and p.suffix in {".py", ".md", ".sh", ".txt", ".toml", ".ipynb", ".json"} \
                and ".ipynb_checkpoints" not in p.parts and p.name != Path(__file__).name:
            text = p.read_text(errors="ignore")
            if "/Users/" in text or "/Volumes/" in text:
                offenders.append(str(p.relative_to(root)))
    assert offenders == []


def test_paths_follow_environment(monkeypatch, tmp_path):
    import importlib

    from ber import config
    monkeypatch.setenv("BER_DATA_DIR", str(tmp_path / "d"))
    monkeypatch.setenv("BER_PERSIST_DIR", str(tmp_path / "p"))
    monkeypatch.setenv("BER_OUTPUT_DIR", str(tmp_path / "o"))
    try:
        c = importlib.reload(config)
        assert c.CLEAN_DIR == tmp_path / "d" / "clean" and c.CAND_DIR == tmp_path / "d" / "cand"
        assert c.MODELS_DIR == tmp_path / "p" / "models" and c.OUTPUT_DIR == tmp_path / "o"
    finally:
        monkeypatch.undo()
        importlib.reload(config)


if __name__ == "__main__":
    pytest.main([__file__])
