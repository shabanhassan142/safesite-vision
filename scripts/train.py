"""
Module 2: Training
Reads configs/train.yaml, runs Ultralytics YOLO training, logs results to
reports/experiments.csv.

Usage:
    python scripts/train.py
    python scripts/train.py --config configs/train.yaml
    python scripts/train.py --config configs/train.yaml --experiment_id E2 --model yolo11s.pt --epochs 80
"""

from __future__ import annotations

import argparse
import csv
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).parent.parent
DEFAULT_CFG = ROOT / "configs" / "train.yaml"
REPORTS_DIR = ROOT / "reports"
EXPERIMENTS_CSV = REPORTS_DIR / "experiments.csv"

CSV_COLUMNS = [
    "experiment_id", "model", "imgsz", "epochs_run",
    "map50", "map50_95", "precision", "recall",
    "train_time_s", "timestamp",
]


# ── Config dataclass ──────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    experiment_id: str
    model: str
    data: str
    imgsz: int
    epochs: int
    batch: int
    seed: int
    project: str
    name: str
    mosaic: float
    hsv_h: float
    hsv_s: float
    hsv_v: float
    fliplr: float
    erasing: float
    patience: int
    close_mosaic: int
    workers: int
    # optional augmentation fields with defaults
    flipud: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 0.0
    perspective: float = 0.0
    degrees: float = 0.0
    mixup: float = 0.0
    copy_paste: float = 0.0

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError(f"epochs must be > 0, got {self.epochs}")
        if self.batch <= 0:
            raise ValueError(f"batch must be > 0, got {self.batch}")
        if self.mosaic > 0.3:
            raise ValueError(
                f"mosaic must be <= 0.3 (training data is already augmented), got {self.mosaic}"
            )
        if self.erasing != 0.0:
            raise ValueError(
                f"erasing (cutout) must be 0.0 (training data already has cutout patches), "
                f"got {self.erasing}"
            )


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_config(path: Path | str, overrides: dict[str, Any] | None = None) -> TrainConfig:
    """Load TrainConfig from a YAML file, optionally applying overrides."""
    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f)
    if overrides:
        raw.update({k: v for k, v in overrides.items() if v is not None})
    # Keep only fields TrainConfig knows about
    known = {f.name for f in TrainConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    filtered = {k: v for k, v in raw.items() if k in known}
    return TrainConfig(**filtered)


# ── Ultralytics kwargs builder ─────────────────────────────────────────────────

def build_args(cfg: TrainConfig) -> dict[str, Any]:
    """Translate TrainConfig into kwargs accepted by YOLO.train()."""
    return {
        "data":         cfg.data,
        "imgsz":        cfg.imgsz,
        "epochs":       cfg.epochs,
        "batch":        cfg.batch,
        "seed":         cfg.seed,
        "project":      cfg.project,
        "name":         cfg.name,
        "patience":     cfg.patience,
        "close_mosaic": cfg.close_mosaic,
        "workers":      cfg.workers,
        "device":       "cpu",
        # augmentation
        "mosaic":       cfg.mosaic,
        "erasing":      cfg.erasing,
        "hsv_h":        cfg.hsv_h,
        "hsv_s":        cfg.hsv_s,
        "hsv_v":        cfg.hsv_v,
        "fliplr":       cfg.fliplr,
        "flipud":       cfg.flipud,
        "translate":    cfg.translate,
        "scale":        cfg.scale,
        "shear":        cfg.shear,
        "perspective":  cfg.perspective,
        "degrees":      cfg.degrees,
        "mixup":        cfg.mixup,
        "copy_paste":   cfg.copy_paste,
        # reproducibility / housekeeping
        "exist_ok":     True,
        "verbose":      True,
    }


# ── CSV logging ───────────────────────────────────────────────────────────────

def log_experiment(
    csv_path: Path,
    experiment_id: str,
    model: str,
    imgsz: int,
    map50: float,
    map50_95: float,
    precision: float,
    recall: float,
    train_time_s: float,
    epochs_run: int = 0,
) -> None:
    """Append one row to the experiments CSV, creating the file if needed."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "experiment_id": experiment_id,
            "model":         model,
            "imgsz":         imgsz,
            "epochs_run":    epochs_run,
            "map50":         round(map50, 4),
            "map50_95":      round(map50_95, 4),
            "precision":     round(precision, 4),
            "recall":        round(recall, 4),
            "train_time_s":  round(train_time_s, 1),
            "timestamp":     datetime.now(timezone.utc).isoformat(),
        })


# ── Training entry point ──────────────────────────────────────────────────────

def run_training(cfg: TrainConfig):
    """Run YOLO training and return the Ultralytics results object."""
    from ultralytics import YOLO

    model = YOLO(cfg.model)
    kwargs = build_args(cfg)

    start = time.time()
    results = model.train(**kwargs)
    elapsed = time.time() - start

    # Extract metrics — Ultralytics stores them on results.results_dict
    rd = getattr(results, "results_dict", {}) or {}
    map50    = float(rd.get("metrics/mAP50(B)",    0.0))
    map50_95 = float(rd.get("metrics/mAP50-95(B)", 0.0))
    precision = float(rd.get("metrics/precision(B)", 0.0))
    recall    = float(rd.get("metrics/recall(B)",    0.0))
    epochs_run = int(getattr(results, "epoch", cfg.epochs) or cfg.epochs)

    log_experiment(
        EXPERIMENTS_CSV,
        cfg.experiment_id, cfg.model, cfg.imgsz,
        map50, map50_95, precision, recall, elapsed, epochs_run,
    )

    # Copy config snapshot next to best.pt for reproducibility
    run_dir = Path(cfg.project) / cfg.name
    best_pt_candidates = list(run_dir.rglob("best.pt"))
    if best_pt_candidates:
        weights_dir = best_pt_candidates[0].parent
        shutil.copy2(DEFAULT_CFG, weights_dir / "train.yaml")

    print(f"\nExperiment {cfg.experiment_id} finished in {elapsed:.0f}s")
    print(f"  mAP50={map50:.4f}  mAP50-95={map50_95:.4f}  "
          f"precision={precision:.4f}  recall={recall:.4f}")
    print(f"  Results logged to: {EXPERIMENTS_CSV}")

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLO model for safesite-vision")
    p.add_argument("--config", default=str(DEFAULT_CFG), help="Path to train.yaml")
    p.add_argument("--experiment_id", default=None)
    p.add_argument("--model",         default=None)
    p.add_argument("--imgsz",         default=None, type=int)
    p.add_argument("--epochs",        default=None, type=int)
    p.add_argument("--batch",         default=None, type=int)
    p.add_argument("--name",          default=None)
    p.add_argument("--mosaic",        default=None, type=float)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    overrides = {k: v for k, v in vars(args).items() if k != "config"}
    cfg = load_config(args.config, overrides)
    print(f"Starting experiment {cfg.experiment_id}: {cfg.model} imgsz={cfg.imgsz} "
          f"epochs={cfg.epochs} batch={cfg.batch}")
    run_training(cfg)


if __name__ == "__main__":
    main()
