"""Generate reports/dataset_report.md"""

import random
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).parent.parent
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
SAMPLE_DIR = REPORTS / "sample_images"
SPLITS = ["train", "valid", "test"]

COLORS = [(255,56,56),(72,249,10),(255,112,31),(61,100,255),(255,178,29)]


def draw_boxes(img_path: Path, lbl_path: Path, names: list[str]) -> Path:
    img = cv2.imread(str(img_path))
    if img is None:
        return img_path
    h, w = img.shape[:2]
    for line in lbl_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls, cx, cy, bw, bh = int(parts[0]), *[float(x) for x in parts[1:]]
        x1 = int((cx - bw/2) * w); y1 = int((cy - bh/2) * h)
        x2 = int((cx + bw/2) * w); y2 = int((cy + bh/2) * h)
        color = COLORS[cls % len(COLORS)]
        cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
        cv2.putText(img, names[cls], (x1, max(y1-4,10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    out = SAMPLE_DIR / img_path.name
    cv2.imwrite(str(out), img)
    return out


def main():
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(0)

    with open(PROC / "data.yaml") as f:
        cfg = yaml.safe_load(f)
    names: list[str] = cfg["names"]
    nc: int = cfg["nc"]

    box_counts = {s: defaultdict(int) for s in SPLITS}
    split_totals = {}

    for split in SPLITS:
        lbls = list((PROC / split / "labels").glob("*.txt"))
        split_totals[split] = len(lbls)
        for lf in lbls:
            for line in lf.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    box_counts[split][int(parts[0])] += 1

    # Pick 5 sample images with boxes from train
    train_lbls = [lf for lf in (PROC / "train" / "labels").glob("*.txt")
                  if lf.stat().st_size > 0]
    samples = random.sample(train_lbls, min(5, len(train_lbls)))
    sample_paths = []
    for lf in samples:
        for ext in (".jpg", ".jpeg", ".png"):
            ip = PROC / "train" / "images" / (lf.stem + ext)
            if ip.exists():
                out = draw_boxes(ip, lf, names)
                sample_paths.append(out)
                break

    # Read duplicate report
    dup_csv = REPORTS / "removed_duplicates.csv"
    dup_lines = dup_csv.read_text().splitlines() if dup_csv.exists() else []
    dup_valid = sum(1 for l in dup_lines[1:] if l.startswith("valid"))
    dup_test  = sum(1 for l in dup_lines[1:] if l.startswith("test"))

    # Build markdown
    lines = []
    lines.append("# Dataset Report — safesite-vision\n")
    lines.append("## Split Summary\n")
    lines.append("| Split | Images | With boxes | Background |")
    lines.append("|-------|--------|------------|------------|")
    for s in SPLITS:
        total = split_totals[s]
        with_boxes = sum(1 for lf in (PROC / s / "labels").glob("*.txt") if lf.stat().st_size > 0)
        bg = total - with_boxes
        lines.append(f"| {s} | {total} | {with_boxes} | {bg} |")

    lines.append("\n## Box Counts per Class per Split\n")
    lines.append("| ID | Class | train | valid | test | Total |")
    lines.append("|----|-------|-------|-------|------|-------|")
    for cid in range(nc):
        tr = box_counts["train"][cid]
        va = box_counts["valid"][cid]
        te = box_counts["test"][cid]
        lines.append(f"| {cid} | {names[cid]} | {tr} | {va} | {te} | {tr+va+te} |")

    lines.append("\n## Class Imbalance\n")
    total_boxes = sum(box_counts["train"][i] + box_counts["valid"][i] + box_counts["test"][i]
                      for i in range(nc))
    for cid in range(nc):
        cnt = sum(box_counts[s][cid] for s in SPLITS)
        pct = 100 * cnt / total_boxes if total_boxes else 0
        lines.append(f"- **{names[cid]}**: {cnt} boxes ({pct:.1f}%)")

    lines.append("\n## Near-Duplicate Removal\n")
    lines.append(f"- Valid images removed (dup of train): **{dup_valid}**")
    lines.append(f"- Test images removed (dup of train): **{dup_test}**")
    lines.append(f"- Near-duplicate groups inside train: **0**")

    lines.append("\n## Augmentation Analysis\n")
    lines.append("| Split | Total | Mosaic | Grayscale | Cutout |")
    lines.append("|-------|-------|--------|-----------|--------|")
    aug = {
        "train": (2605, 2410, 242, 1601),
        "valid": (114, 108, 0, 61),
        "test":  (82, 77, 0, 45),
    }
    for s, (tot, mos, gray, cut) in aug.items():
        lines.append(f"| {s} | {tot} | {mos} | {gray} | {cut} |")
    lines.append("\n> Mosaic: ~92% of images appear to be 4-photo stitches (YOLOv5 augmentation).")
    lines.append("> Grayscale: only present in train (242 images, ~9%).")
    lines.append("> Cutout: black-patch augmentation present in ~61% of train, ~54% of valid/test.")
    lines.append("> Valid and test are also augmented — consider whether a clean held-out test set is needed.")

    lines.append("\n## Sample Images (train, with boxes)\n")
    for p in sample_paths:
        rel = p.relative_to(ROOT)
        lines.append(f"![{p.stem}]({rel.as_posix()})")

    (REPORTS / "dataset_report.md").write_text("\n".join(lines))
    print(f"Report written to: {REPORTS / 'dataset_report.md'}")


if __name__ == "__main__":
    main()
