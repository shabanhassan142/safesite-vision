"""
Tests for Module 2: Training.

Covers:
  - configs/train.yaml loads and contains all required keys with valid values
  - TrainConfig dataclass validates correctly and rejects bad values
  - train.py build_args() produces the right kwargs dict for Ultralytics
  - Augmentation constraints: mosaic <= 0.3, no erasing (cutout), hsv_s > 0, fliplr > 0
  - 1-epoch smoke test on a 20-image fixture dataset (CPU, ~under 2 min)
  - log_experiment() appends a well-formed row to reports/experiments.csv
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

# Make scripts importable
SCRIPTS = Path(__file__).parent.parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

import train as train_module
from train import TrainConfig, build_args, load_config, log_experiment

CONFIGS = Path(__file__).parent.parent.parent / "configs"
TRAIN_YAML = CONFIGS / "train.yaml"

# ── required top-level keys ───────────────────────────────────────────────────

REQUIRED_KEYS = {
    "experiment_id", "model", "data", "imgsz", "epochs", "batch",
    "seed", "project", "name",
    "mosaic", "hsv_h", "hsv_s", "hsv_v", "fliplr", "erasing",
    "patience", "close_mosaic", "workers",
}


class TestTrainYaml:
    def test_file_exists(self):
        assert TRAIN_YAML.exists(), "configs/train.yaml not found"

    def test_required_keys_present(self):
        cfg = yaml.safe_load(TRAIN_YAML.read_text())
        missing = REQUIRED_KEYS - set(cfg.keys())
        assert not missing, f"Missing keys in train.yaml: {missing}"

    def test_mosaic_constraint(self):
        cfg = yaml.safe_load(TRAIN_YAML.read_text())
        assert cfg["mosaic"] <= 0.3, "mosaic must be <= 0.3 (data already augmented)"

    def test_no_extra_cutout(self):
        cfg = yaml.safe_load(TRAIN_YAML.read_text())
        assert cfg["erasing"] == 0.0, "erasing (cutout) must be 0.0 (data already has cutout)"

    def test_hsv_and_fliplr_kept(self):
        cfg = yaml.safe_load(TRAIN_YAML.read_text())
        assert cfg["hsv_s"] > 0, "hsv_s should be > 0"
        assert cfg["hsv_v"] > 0, "hsv_v should be > 0"
        assert cfg["fliplr"] > 0, "fliplr should be > 0"

    def test_epochs_positive(self):
        cfg = yaml.safe_load(TRAIN_YAML.read_text())
        assert cfg["epochs"] > 0

    def test_batch_positive(self):
        cfg = yaml.safe_load(TRAIN_YAML.read_text())
        assert cfg["batch"] > 0


# ── TrainConfig dataclass ─────────────────────────────────────────────────────

class TestTrainConfig:
    def _base(self, **overrides) -> dict:
        base = {
            "experiment_id": "E1",
            "model": "yolo11n.pt",
            "data": "data/processed/data.yaml",
            "imgsz": 640,
            "epochs": 10,
            "batch": 16,
            "seed": 42,
            "project": "models",
            "name": "E1",
            "mosaic": 0.0,
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "fliplr": 0.5,
            "erasing": 0.0,
            "patience": 20,
            "close_mosaic": 10,
            "workers": 4,
        }
        base.update(overrides)
        return base

    def test_valid_config_loads(self):
        cfg = TrainConfig(**self._base())
        assert cfg.experiment_id == "E1"
        assert cfg.model == "yolo11n.pt"

    def test_mosaic_too_high_raises(self):
        with pytest.raises(ValueError, match="mosaic"):
            TrainConfig(**self._base(mosaic=0.5))

    def test_erasing_nonzero_raises(self):
        with pytest.raises(ValueError, match="erasing"):
            TrainConfig(**self._base(erasing=0.4))

    def test_epochs_zero_raises(self):
        with pytest.raises(ValueError, match="epochs"):
            TrainConfig(**self._base(epochs=0))

    def test_batch_zero_raises(self):
        with pytest.raises(ValueError, match="batch"):
            TrainConfig(**self._base(batch=0))


# ── build_args ────────────────────────────────────────────────────────────────

class TestBuildArgs:
    def test_returns_dict_with_required_ultralytics_keys(self):
        cfg = TrainConfig(
            experiment_id="E1", model="yolo11n.pt", data="data.yaml",
            imgsz=640, epochs=1, batch=4, seed=42,
            project="models", name="E1",
            mosaic=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
            fliplr=0.5, erasing=0.0, patience=20, close_mosaic=10, workers=0,
        )
        args = build_args(cfg)
        for key in ("data", "imgsz", "epochs", "batch", "seed", "project",
                    "name", "mosaic", "hsv_s", "fliplr", "erasing"):
            assert key in args, f"'{key}' missing from build_args output"

    def test_device_is_cpu(self):
        cfg = TrainConfig(
            experiment_id="E1", model="yolo11n.pt", data="data.yaml",
            imgsz=640, epochs=1, batch=4, seed=42,
            project="models", name="E1",
            mosaic=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
            fliplr=0.5, erasing=0.0, patience=20, close_mosaic=10, workers=0,
        )
        args = build_args(cfg)
        assert args.get("device") == "cpu"

    def test_values_pass_through(self):
        cfg = TrainConfig(
            experiment_id="E2", model="yolo11s.pt", data="x.yaml",
            imgsz=480, epochs=5, batch=8, seed=0,
            project="models", name="E2",
            mosaic=0.3, hsv_h=0.01, hsv_s=0.5, hsv_v=0.3,
            fliplr=0.5, erasing=0.0, patience=10, close_mosaic=5, workers=0,
        )
        args = build_args(cfg)
        assert args["imgsz"] == 480
        assert args["mosaic"] == 0.3
        assert args["epochs"] == 5


# ── log_experiment ────────────────────────────────────────────────────────────

class TestLogExperiment:
    def test_creates_file_with_header(self, tmp_path):
        csv_path = tmp_path / "experiments.csv"
        log_experiment(csv_path, "E1", "yolo11n", 640, 0.82, 0.79, 0.85, 0.80, 120.5)
        assert csv_path.exists()
        rows = list(csv.DictReader(csv_path.read_text().splitlines()))
        assert len(rows) == 1
        assert rows[0]["experiment_id"] == "E1"

    def test_appends_rows(self, tmp_path):
        csv_path = tmp_path / "experiments.csv"
        log_experiment(csv_path, "E1", "yolo11n", 640, 0.82, 0.79, 0.85, 0.80, 120.5)
        log_experiment(csv_path, "E2", "yolo11s", 640, 0.85, 0.82, 0.87, 0.83, 200.0)
        rows = list(csv.DictReader(csv_path.read_text().splitlines()))
        assert len(rows) == 2
        assert rows[1]["experiment_id"] == "E2"

    def test_required_columns_present(self, tmp_path):
        csv_path = tmp_path / "experiments.csv"
        log_experiment(csv_path, "E1", "yolo11n", 640, 0.82, 0.79, 0.85, 0.80, 120.5)
        rows = list(csv.DictReader(csv_path.read_text().splitlines()))
        for col in ("experiment_id", "model", "imgsz", "map50", "map50_95",
                    "precision", "recall", "train_time_s", "timestamp"):
            assert col in rows[0], f"Column '{col}' missing"


# ── smoke test: 1 epoch on fixture dataset ────────────────────────────────────

@pytest.fixture()
def fixture_dataset(tmp_path: Path) -> Path:
    """Create a minimal YOLO dataset: 20 train + 5 val images, 5 classes."""
    import cv2

    nc = 5
    names = ["person", "hardhat", "no_hardhat", "safety_vest", "no_safety_vest"]

    for split, n in [("train", 20), ("valid", 5)]:
        img_dir = tmp_path / split / "images"
        lbl_dir = tmp_path / split / "labels"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)
        for i in range(n):
            img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            cv2.imwrite(str(img_dir / f"img_{i:04d}.jpg"), img)
            cls = i % nc
            (lbl_dir / f"img_{i:04d}.txt").write_text(
                f"{cls} 0.5 0.5 0.3 0.3\n"
            )

    data_yaml = {
        "path": str(tmp_path),
        "train": "train/images",
        "val": "valid/images",
        "nc": nc,
        "names": names,
    }
    with open(tmp_path / "data.yaml", "w") as f:
        yaml.dump(data_yaml, f)

    return tmp_path


@pytest.mark.smoke
def test_one_epoch_smoke(fixture_dataset: Path, tmp_path: Path):
    """
    Run 1 epoch on the 20-image fixture.  Must complete in under 2 minutes
    and produce a best.pt weights file.
    """
    cfg = TrainConfig(
        experiment_id="smoke",
        model="yolo11n.pt",
        data=str(fixture_dataset / "data.yaml"),
        imgsz=320,
        epochs=1,
        batch=4,
        seed=42,
        project=str(tmp_path / "runs"),
        name="smoke",
        mosaic=0.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        erasing=0.0,
        patience=1,
        close_mosaic=0,
        workers=0,
    )

    start = time.time()
    results = train_module.run_training(cfg)
    elapsed = time.time() - start

    assert elapsed < 120, f"Smoke test took {elapsed:.0f}s, expected < 120s"

    # weights must exist somewhere under project/name
    run_dir = Path(cfg.project) / cfg.name
    weight_candidates = list(run_dir.rglob("best.pt")) + list(run_dir.rglob("last.pt"))
    assert weight_candidates, f"No .pt weights found under {run_dir}"
