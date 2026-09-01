"""
Persiapan dataset: oversampling terarah untuk kelas minoritas.

Masalah yang diselesaikan (dari temuan EDA):
    no-helmet hanya punya 94 instance di train, sementara helmet punya 2116.
    Rasio 22:1 membuat model belajar "hampir semua orang memakai helm" dan
    tetap memperoleh mAP agregat yang terlihat wajar.

Pendekatan:
    Alih-alih menduplikasi file gambar di disk (boros ruang dan berisiko
    mencemari split lain), kita membuat file daftar `train.txt` berisi path
    gambar, di mana gambar yang memuat kelas minoritas dicantumkan beberapa
    kali. Ultralytics menerima file daftar semacam ini pada `data.yaml`.

Aturan keras:
    Oversampling HANYA diterapkan pada split train. Menerapkannya pada valid
    atau test akan mengubah distribusi evaluasi dan membuat metrik tidak lagi
    mencerminkan kondisi nyata.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Set

import yaml

from .config import MINORITY_CLASSES, TrainConfig

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _images_containing(dataset_root: Path, split: str,
                       class_ids: Set[int]) -> List[Path]:
    """Kumpulkan gambar pada split yang memuat minimal satu class_id target."""
    img_dir = dataset_root / split / "images"
    lbl_dir = dataset_root / split / "labels"
    hits: List[Path] = []

    for img in sorted(p for p in img_dir.iterdir()
                      if p.suffix.lower() in IMAGE_EXTS):
        lbl = lbl_dir / f"{img.stem}.txt"
        if not lbl.exists():
            continue
        for line in lbl.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if parts and int(float(parts[0])) in class_ids:
                hits.append(img)
                break
    return hits


def build_oversampled_train_list(cfg: TrainConfig) -> Dict[str, object]:
    """
    Tulis `train_oversampled.txt` dan `data_train.yaml` turunan.

    Return ringkasan untuk dicatat di laporan.
    """
    root = cfg.dataset_root.resolve()
    img_dir = root / "train" / "images"

    all_images = sorted(p for p in img_dir.iterdir()
                        if p.suffix.lower() in IMAGE_EXTS)

    minority_ids = {cfg.class_id(c) for c in MINORITY_CLASSES
                    if c in cfg.class_names}
    minority_images = _images_containing(root, "train", minority_ids)
    minority_set = set(minority_images)

    factor = max(1, int(cfg.oversample_factor))
    lines: List[str] = []
    for img in all_images:
        repeat = factor if img in minority_set else 1
        lines.extend([str(img.resolve())] * repeat)

    list_path = cfg.project_dir / "train_oversampled.txt"
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # data.yaml turunan yang menunjuk ke file daftar untuk train,
    # tetapi tetap memakai folder asli untuk valid dan test.
    derived = {
        "train": str(list_path.resolve()),
        "val": str((root / "valid" / "images").resolve()),
        "test": str((root / "test" / "images").resolve()),
        "nc": len(cfg.class_names),
        "names": cfg.class_names,
    }
    yaml_path = cfg.project_dir / "data_train.yaml"
    yaml_path.write_text(yaml.safe_dump(derived, sort_keys=False),
                         encoding="utf-8")

    summary = {
        "data_yaml": str(yaml_path),
        "train_list": str(list_path),
        "n_original_images": len(all_images),
        "n_minority_images": len(minority_images),
        "oversample_factor": factor,
        "n_effective_samples": len(lines),
        "inflation": round(len(lines) / max(len(all_images), 1), 3),
        "minority_classes": [c for c in MINORITY_CLASSES if c in cfg.class_names],
    }
    logger.info(
        "Oversampling: %d gambar minoritas x%d -> %d sampel efektif (%.2fx)",
        len(minority_images), factor, len(lines), summary["inflation"],
    )
    return summary


def plain_data_yaml(cfg: TrainConfig) -> Path:
    """data.yaml tanpa oversampling, untuk run pembanding."""
    root = cfg.dataset_root.resolve()
    derived = {
        "train": str((root / "train" / "images").resolve()),
        "val": str((root / "valid" / "images").resolve()),
        "test": str((root / "test" / "images").resolve()),
        "nc": len(cfg.class_names),
        "names": cfg.class_names,
    }
    path = cfg.project_dir / "data_plain.yaml"
    path.write_text(yaml.safe_dump(derived, sort_keys=False), encoding="utf-8")
    return path


def prepare(cfg: TrainConfig) -> Dict[str, object]:
    """Titik masuk: siapkan data.yaml sesuai konfigurasi."""
    if cfg.oversample_factor > 1:
        return build_oversampled_train_list(cfg)
    return {
        "data_yaml": str(plain_data_yaml(cfg)),
        "oversample_factor": 1,
        "note": "Oversampling nonaktif.",
    }
