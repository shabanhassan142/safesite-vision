"""
Preview original dataset labels.
- For each original class ID, pick up to 3 random images containing that class,
  draw bounding boxes with class ID + name, save to reports/label_preview/.
- Print a table of box counts per original class ID per split.
"""

import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data" / "raw" / "css-data"
OUT = ROOT / "reports" / "label_preview"

CLASS_NAMES = {
    0: "Hardhat",
    1: "Mask",
    2: "NO-Hardhat",
    3: "NO-Mask",
    4: "NO-Safety Vest",
    5: "Person",
    6: "Safety Cone",
    7: "Safety Vest",
    8: "machinery",
    9: "vehicle",
}

COLORS = [
    (255, 56,  56),   # 0 red
    (255, 157, 151),  # 1 pink
    (255, 112, 31),   # 2 orange
    (255, 178, 29),   # 3 yellow
    (207, 210,  49),  # 4 lime
    (72,  249, 10),   # 5 green
    (146, 234, 43),   # 6 light green
    (61,  694, 255),  # 7 blue  (clamped by cv2)
    (149, 152, 255),  # 8 lavender
    (215, 190, 255),  # 9 purple
]

SPLITS = ["train", "valid", "test"]


def parse_labels(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    boxes = []
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) == 5:
            boxes.append((int(parts[0]), float(parts[1]), float(parts[2]),
                          float(parts[3]), float(parts[4])))
    return boxes


def draw_boxes(img_path: Path, boxes: list, highlight_class: int) -> None:
    img = cv2.imread(str(img_path))
    if img is None:
        return
    h, w = img.shape[:2]
    for cls, cx, cy, bw, bh in boxes:
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        color = COLORS[cls % len(COLORS)]
        thickness = 3 if cls == highlight_class else 1
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        label = f"{cls}:{CLASS_NAMES.get(cls, '?')}"
        cv2.putText(img, label, (x1, max(y1 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    out_name = OUT / f"cls{highlight_class}_{img_path.stem}.jpg"
    cv2.imwrite(str(out_name), img)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    random.seed(42)

    # Collect per-split, per-class: list of (img_path, label_path)
    # and box counts
    class_split_images: dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    box_counts: dict[str, dict[int, int]] = {s: defaultdict(int) for s in SPLITS}

    for split in SPLITS:
        label_dir = DATA / split / "labels"
        image_dir = DATA / split / "images"
        for lf in label_dir.glob("*.txt"):
            boxes = parse_labels(lf)
            # find matching image
            img_path = None
            for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
                candidate = image_dir / (lf.stem + ext)
                if candidate.exists():
                    img_path = candidate
                    break
            if img_path is None:
                continue
            seen_classes = set()
            for cls, *_ in boxes:
                box_counts[split][cls] += 1
                if cls not in seen_classes:
                    class_split_images[cls][split].append((img_path, lf, boxes))
                    seen_classes.add(cls)

    # Draw previews: for each class pick up to 3 images (prefer train, then valid, then test)
    for cls_id in sorted(CLASS_NAMES.keys()):
        candidates = []
        for split in SPLITS:
            candidates.extend(class_split_images[cls_id][split])
        if not candidates:
            print(f"  Class {cls_id} ({CLASS_NAMES[cls_id]}): no images found")
            continue
        sample = random.sample(candidates, min(3, len(candidates)))
        for img_path, lf, boxes in sample:
            draw_boxes(img_path, boxes, cls_id)
        print(f"  Class {cls_id} ({CLASS_NAMES[cls_id]}): saved {len(sample)} preview(s)")

    # Print box count table
    all_ids = sorted(CLASS_NAMES.keys())
    header = f"{'Class ID':<10} {'Name':<18}" + "".join(f"{s:>10}" for s in SPLITS) + f"{'Total':>10}"
    print("\n" + "=" * len(header))
    print("Box counts per original class ID per split")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for cid in all_ids:
        row = f"{cid:<10} {CLASS_NAMES[cid]:<18}"
        total = 0
        for s in SPLITS:
            n = box_counts[s][cid]
            row += f"{n:>10}"
            total += n
        row += f"{total:>10}"
        print(row)
    print("=" * len(header))
    print(f"\nPreview images saved to: {OUT}")


if __name__ == "__main__":
    main()
