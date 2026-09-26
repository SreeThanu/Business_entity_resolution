"""Check the raw dataset: checksums, optional comparison with an old copy, and format checks.

Usage:
    python scripts/00_check_raw.py                      # hash + format-check config.RAW_DATASET_DIR
    python scripts/00_check_raw.py --old-copy PATH      # also compare with PATH/dataset (read-only)
    python scripts/00_check_raw.py --verify             # check config.RAW_DATASET_DIR against the committed hashes

Writes checksums/fresh_dataset.sha256 (and checksums/old_copy_dataset.sha256 when --old-copy is
given) in sha256sum format. With --verify it writes nothing and exits non-zero if any file's hash
differs from checksums/fresh_dataset.sha256. Never writes to the old copy.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import sys
from pathlib import Path

from ber import config

CHUNK = 8 * 1024 * 1024
EXPECTED_FIELDS = {"train_ground_truth.tsv": 2}  # every other TSV has 4


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def dataset_files(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*.tsv"))


def hash_tree(root: Path, out: Path | None, label: str) -> dict[str, str | None]:
    """Hash every TSV under root. A file that fails to read is logged and recorded as None."""
    result: dict[str, str | None] = {}
    lines = []
    for rel in dataset_files(root):
        try:
            digest = sha256_file(root / rel)
            lines.append(f"{digest}  {rel}")
            print(f"[{label}] {rel}: {digest}", flush=True)
        except OSError as e:
            digest = None
            print(f"[{label}] READ FAILED {rel}: {e!r}", flush=True)
        result[rel] = digest
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines) + "\n")
    return result


def verify(fresh: dict[str, str | None], ref_path: Path) -> bool:
    """Compare fresh hashes with a sha256sum-format reference file. True if every file matches."""
    ref = dict(reversed(line.split(maxsplit=1)) for line in ref_path.read_text().splitlines() if line)
    print(f"\nverify against {ref_path}\n| file | result |\n|---|---|")
    ok = True
    for rel in sorted(set(ref) | set(fresh)):
        if rel not in fresh:
            res = "MISSING locally"
        elif rel not in ref:
            res = "NOT IN REFERENCE"
        elif fresh[rel] is None:
            res = "READ FAILED"
        else:
            res = "MATCH" if fresh[rel] == ref[rel] else "DIFFER"
        ok &= res == "MATCH"
        print(f"| {rel} | {res} |")
    print("checksums OK" if ok else "CHECKSUM MISMATCH")
    return ok


def line_diff(fresh: Path, old: Path, max_diffs: int = 5) -> None:
    """Print row counts and the first differing lines of two text files, side by side."""
    try:
        with open(fresh, "rb") as a, open(old, "rb") as b:
            shown, n_a, n_b = 0, 0, 0
            for i, (la, lb) in enumerate(itertools.zip_longest(a, b), start=1):
                n_a += la is not None
                n_b += lb is not None
                if la != lb and shown < max_diffs:
                    print(f"  line {i}:")
                    print(f"    fresh: {la!r}")
                    print(f"    old  : {lb!r}")
                    shown += 1
            print(f"  lines: fresh={n_a:,} old={n_b:,}")
    except OSError as e:
        print(f"  diff aborted, read failed: {e!r}")


def check_format(path: Path, n_fields: int) -> tuple[int, int, int]:
    """Return (data_rows, bad_field_lines, utf8_error_lines). The header is excluded from rows."""
    rows = bad_fields = bad_utf8 = 0
    with open(path, "rb") as f:
        for i, raw in enumerate(f):
            try:
                line = raw.decode("utf-8")
            except UnicodeDecodeError:
                bad_utf8 += 1
                if bad_utf8 <= 3:
                    print(f"  {path.name}:{i + 1} invalid UTF-8: {raw[:120]!r}")
                continue
            line = line.rstrip("\n")
            if line.endswith("\r"):
                print(f"  {path.name}:{i + 1} has CRLF")
            if line.count("\t") != n_fields - 1:
                bad_fields += 1
                if bad_fields <= 3:
                    print(f"  {path.name}:{i + 1} has {line.count(chr(9)) + 1} fields: {line[:120]!r}")
            if i > 0:
                rows += 1
    return rows, bad_fields, bad_utf8


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old-copy", type=Path, help="old student_resource dir (read-only)")
    ap.add_argument("--verify", action="store_true",
                    help="compare the raw dataset with checksums/fresh_dataset.sha256 instead of rewriting it")
    args = ap.parse_args()

    fresh_root = config.RAW_DATASET_DIR
    ref_path = config.CHECKSUM_DIR / "fresh_dataset.sha256"
    fresh = hash_tree(fresh_root, None if args.verify else ref_path, "fresh")
    hashes_ok = verify(fresh, ref_path) if args.verify else True

    if args.old_copy:
        old_root = args.old_copy / "dataset"
        old = hash_tree(old_root, config.CHECKSUM_DIR / "old_copy_dataset.sha256", "old")
        print("\n| file | fresh sha256 | old sha256 | result |\n|---|---|---|---|")
        differ = []
        for rel in sorted(set(fresh) | set(old)):
            a, b = fresh.get(rel), old.get(rel)
            if rel not in old:
                res = "MISSING in old"
            elif b is None:
                res = "OLD READ FAILED (skipped)"
            elif a == b:
                res = "MATCH"
            else:
                res = "DIFFER"
                differ.append(rel)
            print(f"| {rel} | {(a or '-')[:16]}… | {(b or '-')[:16]}… | {res} |")
        for rel in differ:
            print(f"\nDIFF {rel}")
            line_diff(fresh_root / rel, old_root / rel)
        print(f"\ncompared={len(fresh)} differ={len(differ)}")

    print("\n| file | data_rows | bad_field_lines | utf8_error_lines |\n|---|---|---|---|")
    ok = True
    for rel in dataset_files(fresh_root):
        name = Path(rel).name
        rows, bf, bu = check_format(fresh_root / rel, EXPECTED_FIELDS.get(name, 4))
        ok &= bf == 0 and bu == 0
        print(f"| {name} | {rows:,} | {bf} | {bu} |")
    print("format OK" if ok else "FORMAT PROBLEMS FOUND")
    return 0 if ok and hashes_ok else 1


if __name__ == "__main__":
    sys.exit(main())
