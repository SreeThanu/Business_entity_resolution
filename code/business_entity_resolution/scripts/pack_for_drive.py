"""List the data files the Colab pipeline (07-10) needs and write their manifest.

    python scripts/pack_for_drive.py                       # hash the files, write colab/data_manifest.json
    python scripts/pack_for_drive.py --copy-to <folder>    # also copy them there (e.g. a Google Drive
                                                           # for desktop folder), keeping the data/ layout

The manifest (committed) lists every file with its path relative to the data dir, size and
sha256. On Colab, scripts/fetch_data.py copies exactly these files from Drive to the local disk
and verifies them before anything runs.

Included: data/clean (Stage 1 + stage2_*), data/splits, data/dicts, the three candidate files
(cand_train_200k, cand_val_50k, test) and the long ground truth (training labels). Blocking caches
(data/cand/cache, data/cand/dev) and raw TSVs are NOT included: 07-10 never read them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

from ber import config

MANIFEST = Path(__file__).resolve().parents[1] / "colab" / "data_manifest.json"


def needed_files() -> list[Path]:
    d = config.DATA_DIR
    files = [config.clean_source_path(split, s) for split in config.SPLITS for s in config.SOURCES]
    files += [config.stage2_source_path(split, s) for split in config.SPLITS for s in config.SOURCES]
    files += sorted(p for p in config.SPLITS_DIR.iterdir() if p.is_file() and not p.name.startswith("."))
    files += sorted(config.DICTS_DIR.glob("*.parquet"))
    files += [config.cand_path(n) for n in ("cand_train_200k", "cand_val_50k", "test")]
    files += [config.parquet_ground_truth_path(long=True)]
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        sys.exit(f"missing: {missing}")
    for f in files:
        f.relative_to(d)  # everything must live under the data dir
    return files


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.2f} {unit}" if unit == "GB" else f"{n:.0f} {unit}"
        n /= 1024
    return str(n)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--copy-to", type=Path, default=None, help="copy the files into this folder (data/ layout)")
    args = ap.parse_args()

    entries = []
    t = time.time()
    for f in needed_files():
        rel = f.relative_to(config.DATA_DIR).as_posix()
        entries.append({"path": rel, "bytes": f.stat().st_size, "sha256": sha256(f)})
        print(f"{human(entries[-1]['bytes']):>10}  {rel}")
    total = sum(e["bytes"] for e in entries)
    manifest = {"description": "Data needed by scripts/07-10 (Colab). Paths are relative to BER_DATA_DIR "
                               "(Drive: the shared ber_data folder). Written by scripts/pack_for_drive.py.",
                "created": time.strftime("%Y-%m-%d"), "files": len(entries), "total_bytes": total,
                "total_human": human(total), "entries": entries}
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"\n{len(entries)} files, total {human(total)} ({total:,} bytes); hashed in {time.time() - t:.0f}s")
    print(f"manifest -> {MANIFEST}")

    if args.copy_to:
        for e in entries:
            dst = args.copy_to / e["path"]
            if dst.exists() and dst.stat().st_size == e["bytes"]:
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(config.DATA_DIR / e["path"], dst)
            print(f"copied {e['path']}")
        print(f"copied to {args.copy_to}; upload that folder's contents as the shared Drive folder ber_data/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
