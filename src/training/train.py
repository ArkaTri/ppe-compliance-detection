"""
Training: pembungkus Ultralytics dengan parameter eksplisit dan pencatatan
provenance.

Dua hal yang ditambahkan di atas `model.train()` biasa:

  1. Seluruh parameter augmentasi dikirim EKSPLISIT. Default Ultralytics
     mengaktifkan `fliplr=0.5` dan `mosaic=1.0`; keduanya geometris dan
     dilarang panduan capstone.
  2. Konfigurasi lengkap disimpan ke JSON di direktori run. Tanpa ini,
     tiga bulan kemudian tidak ada cara memastikan bobot mana dilatih
     dengan pengaturan apa.
"""

from __future__ import annotations

import json
import logging
import platform
import time
from pathlib import Path
from typing import Dict

from .config import TrainConfig
from . import dataset_prep

logger = logging.getLogger(__name__)


def _environment() -> Dict[str, str]:
    info = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = str(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        info["torch"] = "tidak terdeteksi"
    try:
        import ultralytics
        info["ultralytics"] = ultralytics.__version__
    except Exception:  # noqa: BLE001
        info["ultralytics"] = "tidak terdeteksi"
    return info


def train(cfg: TrainConfig) -> Dict[str, object]:
    """Jalankan satu run training dan kembalikan ringkasannya."""
    from ultralytics import YOLO

    prep = dataset_prep.prepare(cfg)
    data_yaml = prep["data_yaml"]

    logger.info("Run '%s' | imgsz=%d | aug=%s | oversample=x%s",
                cfg.run_name, cfg.imgsz, cfg.aug_policy,
                prep.get("oversample_factor"))

    model = YOLO(cfg.weights)
    started = time.time()

    model.train(
        data=data_yaml,
        epochs=cfg.epochs,
        batch=cfg.batch,
        imgsz=cfg.imgsz,
        patience=cfg.patience,
        device=cfg.device or None,
        workers=cfg.workers,
        seed=cfg.seed,
        project=str(cfg.project_dir),
        name=cfg.run_name,
        exist_ok=True,
        val=True,
        plots=True,
        deterministic=True,
        **cfg.augmentation,
    )

    elapsed = time.time() - started

    manifest = {
        "config": cfg.to_dict(),
        "dataset_prep": prep,
        "environment": _environment(),
        "elapsed_seconds": round(elapsed, 1),
        "best_weights": str(cfg.best_weights),
    }
    (cfg.run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    logger.info("Selesai dalam %.1f menit. Bobot: %s",
                elapsed / 60, cfg.best_weights)
    return manifest
