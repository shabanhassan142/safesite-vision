"""
Apply class mapping to raw dataset and write data/processed/.
Mapping is read from configs/dataset.yaml.
Images that become empty after dropping boxes are kept as background (empty label file).
"""

import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent
CFG_PATH = ROOT / "configs" / "dataset.yaml"
SPLITS = ["train", "valid", "test"]


def load_config() -> dict:
    with open(CFG_PATH) as f:
        return yaml.safe_load(f)


def remap_label_file(src: Path, mapping: dict[int, int]) -> list[str]:
    """Return remapped YOLO label lines; drop boxes not in mapping."""
    lines = []
    for line in src.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        old_id = int(parts[0])
        if old_id in mapping:
            lines.append(f"{mapping[old_id]} {' '.join(parts[1:])}")
    return lines


def main() -> None:
    cfg = load_config()
    src_root = ROOT / cfg["source_data"]
    dst_root = ROOT / cfg["processed_data"]
    mapping: dict[int, int] = {int(k): int(v) for k, v in cfg["class_mapping"].items()}
    class_names: list[str] = cfg["class_names"]

    counts = {s: {"total": 0, "background": 0, "kept": 0} for s in SPLITS}

    for split in SPLITS:
        src_img = src_root / split / "images"
        src_lbl = src_root / split / "labels"
        dst_img = dst_root / split / "images"
        dst_lbl = dst_root / split / "labels"
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)

        for lf in sorted(src_lbl.glob("*.txt")):
            # find matching image
            img_path = None
            for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
                c = src_img / (lf.stem + ext)
                if c.exists():
                    img_path = c
                    break
            if img_path is None:
                continue

            new_lines = remap_label_file(lf, mapping)
            counts[split]["total"] += 1

            # copy image
            shutil.copy2(img_path, dst_img / img_path.name)

            # write label (empty = background)
            out_lbl = dst_lbl / lf.name
            out_lbl.write_text("\n".join(new_lines))

            if new_lines:
                counts[split]["kept"] += 1
            else:
                counts[split]["background"] += 1

    # Write data.yaml
    data_yaml = {
        "path": str(dst_root.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(class_names),
        "names": class_names,
    }
    with open(dst_root / "data.yaml", "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

    print("\nDataset preparation complete")
    print(f"{'Split':<8} {'Total':>8} {'With boxes':>12} {'Background':>12}")
    print("-" * 44)
    for s in SPLITS:
        c = counts[s]
        print(f"{s:<8} {c['total']:>8} {c['kept']:>12} {c['background']:>12}")
    print(f"\ndata.yaml written to: {dst_root / 'data.yaml'}")


if __name__ == "__main__":
    main()
