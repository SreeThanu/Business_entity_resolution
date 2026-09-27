"""Copy the pipeline's data from Google Drive to the local disk and verify it (Colab step 4).

    python scripts/fetch_data.py --src /content/drive/MyDrive/ber_data --dst /content/ber_data

Copies exactly the files in colab/data_manifest.json (written by scripts/pack_for_drive.py) and
checks every sha256 and size. A file already at --dst with the right checksum is not copied
again, so this is safe to re-run. Exits non-zero, naming the files, if anything is missing or
corrupt; nothing downstream should run until it prints "ALL FILES VERIFIED".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MANIFEST = Path(__file__).resolve().parents[1] / "colab" / "data_manifest.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def fetch(e: dict, src: Path, dst: Path) -> str | None:
    """Returns an error message, or None if the file at dst is verified."""
    target = dst / e["path"]
    if target.exists() and target.stat().st_size == e["bytes"] and sha256(target) == e["sha256"]:
        return None
    source = src / e["path"]
    if not source.exists():
        return f"{e['path']}: not found in {src}"
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    t = time.time()
    shutil.copyfile(source, tmp)
    if tmp.stat().st_size != e["bytes"]:
        return f"{e['path']}: size {tmp.stat().st_size:,} != manifest {e['bytes']:,}"
    if sha256(tmp) != e["sha256"]:
        return f"{e['path']}: sha256 mismatch (file on Drive differs from the manifest)"
    tmp.replace(target)
    print(f"  copied + verified {e['path']} ({e['bytes'] / 2**20:,.0f} MB, {time.time() - t:.0f}s)", flush=True)
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="Drive folder with the data/ layout (ber_data)")
    ap.add_argument("--dst", type=Path, required=True, help="local data dir (BER_DATA_DIR)")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    m = json.loads(MANIFEST.read_text())
    print(f"{m['files']} files, {m['total_human']} (manifest {m['created']}): {args.src} -> {args.dst}", flush=True)
    t = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        errors = [r for r in ex.map(lambda e: fetch(e, args.src, args.dst), m["entries"]) if r]
    if errors:
        print("FAILED:\n  " + "\n  ".join(errors))
        return 1
    print(f"ALL FILES VERIFIED ({m['files']} files, {m['total_human']}, {time.time() - t:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
