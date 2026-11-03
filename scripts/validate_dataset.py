"""
Validate data/processed dataset.
Checks:
  - Label format: each line has exactly 5 fields
  - Box values cx, cy, w, h are in (0, 1]
  - Class IDs are valid (0 to nc-1)
  - No image filename appears in more than one split
  - Prints box counts per class per split
Exits with code 1 if any error is found.
"""

import sys
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent
PROC = ROOT / "data" / "processed"
SPLITS = ["train", "valid", "test"]


def load_nc() -> tuple[int, list[str]]:
    with open(PROC / "data.yaml") as f:
        cfg = yaml.safe_load(f)
    return cfg["nc"], cfg["names"]


def validate(nc: int, names: list[str]) -> bool:
    errors: list[str] = []
    all_images: dict[str, str] = {}   # filename -> split
    box_counts: dict[str, dict[int, int]] = {s: defaultdict(int) for s in SPLITS}

    for split in SPLITS:
        lbl_dir = PROC / split / "labels"
        img_dir = PROC / split / "images"

        for lf in sorted(lbl_dir.glob("*.txt")):
            # check image exists
            found_img = None
            for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
                c = img_dir / (lf.stem + ext)
                if c.exists():
                    found_img = c
                    break
            if found_img is None:
                errors.append(f"[{split}] No image for label: {lf.name}")
                continue

            # cross-split duplicate check
            if found_img.name in all_images:
                errors.append(
                    f"Image appears in multiple splits: {found_img.name} "
                    f"({all_images[found_img.name]} and {split})"
                )
            else:
                all_images[found_img.name] = split

            # parse label
            for lineno, line in enumerate(lf.read_text().splitlines(), 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    errors.append(
                        f"[{split}] {lf.name}:{lineno} — expected 5 fields, got {len(parts)}"
                    )
                    continue
                try:
                    cls = int(parts[0])
                    vals = [float(x) for x in parts[1:]]
                except ValueError:
                    errors.append(f"[{split}] {lf.name}:{lineno} — non-numeric fields")
                    continue

                if cls < 0 or cls >= nc:
                    errors.append(
                        f"[{split}] {lf.name}:{lineno} — invalid class id {cls} (nc={nc})"
                    )
                for i, v in enumerate(vals):
                    if not (0.0 < v <= 1.0):
                        errors.append(
                            f"[{split}] {lf.name}:{lineno} — field {i+1} out of range: {v}"
                        )

                box_counts[split][cls] += 1

    # ── report ──
    print("\nBox counts per class per split (processed dataset):")
    header = f"  {'ID':<5} {'Name':<18}" + "".join(f"{s:>8}" for s in SPLITS) + f"{'Total':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for cid in range(nc):
        row = f"  {cid:<5} {names[cid]:<18}"
        total = 0
        for s in SPLITS:
            n = box_counts[s][cid]
            row += f"{n:>8}"
            total += n
        row += f"{total:>8}"
        print(row)

    if errors:
        print(f"\n[FAIL] {len(errors)} error(s) found:")
        for e in errors[:20]:
            print(f"  {e}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")
        return False

    print(f"\n[PASS] All checks passed.")
    return True


def main() -> None:
    nc, names = load_nc()
    ok = validate(nc, names)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
