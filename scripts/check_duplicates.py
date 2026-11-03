"""
Step 3: Near-duplicate detection + augmentation analysis.
- Compute perceptual hashes for all images in data/processed.
- Report valid/test images that are near-duplicates of a train image (hamming <= threshold).
- Remove those from valid/test, log to reports/removed_duplicates.csv.
- Report near-duplicate groups inside train.
- Augmentation analysis per split:
    * mosaic:    image divided into 4 equal quadrants, each quadrant has significantly
                 different mean brightness (simple proxy for 4-photo stitch)
    * grayscale: max channel-diff across all pixels < 10
    * cutout:    at least one 20x20 region has near-black mean (<= 15)
- Save 6 random sample images per split to reports/split_preview/.
"""

import csv
import random
from collections import defaultdict
from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image

ROOT = Path(__file__).parent.parent
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
PREVIEW_DIR = REPORTS / "split_preview"
DUP_CSV = REPORTS / "removed_duplicates.csv"

HASH_SIZE = 8
HAMMING_THRESHOLD = 6   # <= this => near-duplicate
SPLITS = ["train", "valid", "test"]


# ── augmentation detectors ──────────────────────────────────────────────────

def is_grayscale(img_bgr: np.ndarray) -> bool:
    """True if R≈G≈B everywhere (max per-pixel channel spread < 10)."""
    b, g, r = cv2.split(img_bgr)
    diff = np.max([np.abs(r.astype(int) - g),
                   np.abs(r.astype(int) - b),
                   np.abs(g.astype(int) - b)], axis=0)
    return int(diff.max()) < 10


def is_mosaic(img_bgr: np.ndarray) -> bool:
    """True if the 4 quadrant means differ by > 20 in brightness (rough mosaic proxy)."""
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    quads = [
        gray[:h//2, :w//2],
        gray[:h//2, w//2:],
        gray[h//2:, :w//2],
        gray[h//2:, w//2:],
    ]
    means = [q.mean() for q in quads]
    return (max(means) - min(means)) > 20


def has_cutout(img_bgr: np.ndarray, block: int = 20, thresh: int = 15) -> bool:
    """True if any block×block region has mean brightness <= thresh (black patch)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    for y in range(0, h - block, block):
        for x in range(0, w - block, block):
            if gray[y:y+block, x:x+block].mean() <= thresh:
                return True
    return False


# ── hashing ─────────────────────────────────────────────────────────────────

def get_hash(path: Path) -> imagehash.ImageHash:
    return imagehash.phash(Image.open(path), hash_size=HASH_SIZE)


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    random.seed(42)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    # ── collect all image paths ──
    split_images: dict[str, list[Path]] = {}
    for s in SPLITS:
        split_images[s] = sorted((PROC / s / "images").glob("*.*"))
    
    print("Computing perceptual hashes...")
    split_hashes: dict[str, dict[Path, imagehash.ImageHash]] = {}
    for s in SPLITS:
        split_hashes[s] = {}
        for p in split_images[s]:
            try:
                split_hashes[s][p] = get_hash(p)
            except Exception:
                pass
        print(f"  {s}: {len(split_hashes[s])} hashes computed")

    # ── near-dup: valid/test vs train ──
    train_hashes = list(split_hashes["train"].items())  # [(path, hash), ...]

    removed: list[dict] = []
    for s in ["valid", "test"]:
        for p, h in split_hashes[s].items():
            for tp, th in train_hashes:
                if (h - th) <= HAMMING_THRESHOLD:
                    removed.append({
                        "split": s,
                        "image": p.name,
                        "duplicate_of": tp.name,
                        "hamming": h - th,
                    })
                    break  # one match is enough

    print(f"\nNear-duplicates found (valid/test vs train):")
    dup_valid = sum(1 for r in removed if r["split"] == "valid")
    dup_test  = sum(1 for r in removed if r["split"] == "test")
    print(f"  valid: {dup_valid} / {len(split_hashes['valid'])}")
    print(f"  test:  {dup_test}  / {len(split_hashes['test'])}")

    # Write CSV
    with open(DUP_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "image", "duplicate_of", "hamming"])
        w.writeheader()
        w.writerows(removed)

    # Remove from valid/test
    removed_paths = {r["image"] for r in removed}
    for s in ["valid", "test"]:
        for p in list(split_images[s]):
            if p.name in removed_paths:
                # remove image and label
                p.unlink(missing_ok=True)
                lbl = PROC / s / "labels" / (p.stem + ".txt")
                lbl.unlink(missing_ok=True)
    print(f"  Removed files written to: {DUP_CSV}")

    # ── near-dup groups inside train ──
    print("\nFinding near-duplicate groups inside train...")
    train_paths = list(split_hashes["train"].keys())
    train_hash_list = [split_hashes["train"][p] for p in train_paths]
    visited = set()
    groups = []
    for i, (p, h) in enumerate(zip(train_paths, train_hash_list)):
        if i in visited:
            continue
        group = [p]
        for j in range(i + 1, len(train_paths)):
            if j not in visited and (h - train_hash_list[j]) <= HAMMING_THRESHOLD:
                group.append(train_paths[j])
                visited.add(j)
        if len(group) > 1:
            groups.append(group)
        visited.add(i)
    print(f"  Near-duplicate groups in train: {len(groups)}")
    total_dup_train = sum(len(g) - 1 for g in groups)
    print(f"  Excess images (one per group is 'original'): {total_dup_train}")

    # ── augmentation analysis ──
    print("\nAugmentation analysis per split:")
    aug_stats: dict[str, dict] = {}
    for s in SPLITS:
        imgs = split_images[s]
        n = len(imgs)
        mosaic_count = gray_count = cutout_count = 0
        for p in imgs:
            img = cv2.imread(str(p))
            if img is None:
                continue
            if is_mosaic(img):
                mosaic_count += 1
            if is_grayscale(img):
                gray_count += 1
            if has_cutout(img):
                cutout_count += 1
        aug_stats[s] = {
            "total": n,
            "mosaic": mosaic_count,
            "grayscale": gray_count,
            "cutout": cutout_count,
        }
        print(f"  {s:6s}: total={n:5d}  mosaic={mosaic_count:5d}  "
              f"grayscale={gray_count:5d}  cutout={cutout_count:5d}")

    # ── split previews ──
    print("\nSaving split preview images...")
    for s in SPLITS:
        imgs = split_images[s]
        sample = random.sample(imgs, min(6, len(imgs)))
        for i, p in enumerate(sample):
            img = cv2.imread(str(p))
            if img is None:
                continue
            # overlay split name and augmentation flags
            flags = []
            if is_mosaic(img): flags.append("mosaic")
            if is_grayscale(img): flags.append("gray")
            if has_cutout(img): flags.append("cutout")
            label = f"{s} | {','.join(flags) if flags else 'normal'}"
            cv2.putText(img, label, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            out = PREVIEW_DIR / f"{s}_{i+1:02d}_{p.stem}.jpg"
            cv2.imwrite(str(out), img)
    print(f"  Preview images saved to: {PREVIEW_DIR}")

    # ── summary ──
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Near-duplicates removed: valid={dup_valid}, test={dup_test}")
    print(f"Near-dup groups in train: {len(groups)} ({total_dup_train} excess images)")
    print("\nAugmentation report:")
    print(f"  {'Split':<8} {'Total':>7} {'Mosaic':>8} {'Grayscale':>11} {'Cutout':>8}")
    print(f"  {'-'*46}")
    for s in SPLITS:
        st = aug_stats[s]
        print(f"  {s:<8} {st['total']:>7} {st['mosaic']:>8} {st['grayscale']:>11} {st['cutout']:>8}")


if __name__ == "__main__":
    main()
