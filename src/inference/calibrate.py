"""
Kalibrasi ulang ambang keyakinan PADA KONFIGURASI PRODUKSI.

Masalah yang diperbaiki:
    Berkas `evaluation_test.json` dihasilkan oleh pipeline training, yang
    memanggil model secara langsung pada gambar penuh - satu tahap, tanpa
    kaskade. Sapuan kurva F2/F1 yang menghasilkan ambang per kelas dilakukan
    pada distribusi keyakinan itu.

    Produksi menjalankan deteksi kaskade. Deteksi tahap dua berasal dari
    potongan yang diperbesar ke resolusi inferensi penuh, sehingga objek
    kecil - persis kelas yang paling bermasalah - menghasilkan keyakinan
    yang sistematis berbeda.

    Menerapkan ambang yang dioptimalkan pada distribusi A ke distribusi B
    bukan sekadar suboptimal; itu ketidaksesuaian yang tidak menghasilkan
    gejala error apa pun.

Catatan metodologis:
    Ambang `person` TIDAK disapu di sini. Alasannya kausal: ambang person
    menentukan pekerja mana yang dipotong untuk tahap dua, sehingga
    mengubahnya mengubah deteksi APD yang sedang diukur. Menyapu keduanya
    sekaligus akan mencampur sebab dan akibat.
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .config import InferenceConfig, PERSON, PPE_CLASSES
from .evaluate_utils import match_image, metrics_at, sweep_thresholds

logger = logging.getLogger(__name__)

VIOLATION_CLASSES = ("no-helmet", "no-vest")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _load_gt(label_path: Path, w: int, h: int, n_classes: int):
    """Baca label YOLO menjadi (boxes xyxy, class_ids)."""
    if not label_path.exists():
        return np.zeros((0, 4)), np.zeros(0, int)
    boxes, classes = [], []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cid = int(float(parts[0]))
            xc, yc, bw, bh = (float(v) for v in parts[1:5])
        except ValueError:
            continue
        if not 0 <= cid < n_classes:
            continue
        boxes.append([(xc - bw / 2) * w, (yc - bh / 2) * h,
                      (xc + bw / 2) * w, (yc + bh / 2) * h])
        classes.append(cid)
    if not boxes:
        return np.zeros((0, 4)), np.zeros(0, int)
    return np.asarray(boxes, float), np.asarray(classes, int)


def collect_matches_production(dataset_root: Path, split: str,
                               class_names: Sequence[str],
                               cfg: InferenceConfig,
                               limit: Optional[int] = None) -> List[dict]:
    """
    Kumpulkan pencocokan prediksi-ground truth MEMAKAI jalur produksi.

    Berbeda dari `evaluate.collect_matches` yang memanggil model langsung,
    fungsi ini melewati `detect_cascade` sehingga distribusi keyakinan yang
    disapu adalah distribusi yang benar-benar dipakai saat melayani pengguna.
    """
    from PIL import Image
    from .cascade import detect_cascade
    from .pipeline import load_model

    model = load_model(cfg)
    names = list(class_names)

    # Ambang APD diturunkan ke lantai agar seluruh kurva dapat disapu dari
    # satu kali inferensi. Ambang `person` dibiarkan apa adanya.
    sweep_cfg = copy.deepcopy(cfg)
    for name in PPE_CLASSES:
        sweep_cfg.thresholds[name] = 0.01

    img_dir = Path(dataset_root) / split / "images"
    lbl_dir = Path(dataset_root) / split / "labels"
    images = sorted(p for p in img_dir.iterdir()
                    if p.suffix.lower() in IMAGE_EXTS)
    if limit:
        images = images[:limit]

    per_image: List[dict] = []
    for i, img_path in enumerate(images, 1):
        if i % 20 == 0:
            logger.info("  %d/%d gambar", i, len(images))

        with Image.open(img_path) as im:
            arr = np.asarray(im.convert("RGB"))
            w, h = im.size

        dets, _ = detect_cascade(arr, sweep_cfg, model)

        if dets:
            pb = np.array([d.xyxy for d in dets], float)
            pc = np.array([names.index(d.class_name) if d.class_name in names
                           else -1 for d in dets], int)
            pf = np.array([d.confidence for d in dets], float)
            keep = pc >= 0
            pb, pc, pf = pb[keep], pc[keep], pf[keep]
        else:
            pb, pc, pf = np.zeros((0, 4)), np.zeros(0, int), np.zeros(0)

        gb, gc = _load_gt(lbl_dir / f"{img_path.stem}.txt", w, h, len(names))

        per_image.append({
            "image_id": img_path.stem,
            "records": match_image(pb, pc, pf, gb, gc, iou_thr=0.5),
        })

    return per_image


def recalibrate(dataset_root: Path, split: str, class_names: Sequence[str],
                cfg: InferenceConfig, output_path: Optional[Path] = None,
                limit: Optional[int] = None) -> Dict[str, object]:
    """
    Sapu ulang ambang per kelas pada konfigurasi produksi.

    Kelas pelanggaran dioptimalkan pada F2 (recall dibobot dua kali), kelas
    lain pada F1 - kriteria yang sama dengan kalibrasi awal, hanya distribusi
    datanya yang kini benar.
    """
    per_image = collect_matches_production(dataset_root, split, class_names,
                                           cfg, limit=limit)
    names = list(class_names)
    per_class: Dict[str, object] = {}

    for name in names:
        cid = names.index(name)

        if name == PERSON:
            old = cfg.threshold(PERSON)
            m = metrics_at(per_image, cid, old)
            per_class[name] = {
                "recommended_threshold": old,
                "optimized_for": "dipertahankan (menentukan potongan kaskade)",
                "precision": m["precision"], "recall": m["recall"],
                "f1": m["f1"], "f2": m["f2"], "n_gt": m["n_gt"],
            }
            continue

        key = "f2" if name in VIOLATION_CLASSES else "f1"
        sweep = sweep_thresholds(per_image, cid)
        if not sweep:
            continue
        best = max(sweep, key=lambda r: r[key])
        per_class[name] = {
            "recommended_threshold": best["threshold"],
            "optimized_for": key,
            "precision": best["precision"], "recall": best["recall"],
            "f1": best["f1"], "f2": best["f2"], "n_gt": best["n_gt"],
            "previous_threshold": cfg.threshold(name),
            "sweep": sweep,
        }

    out = {
        "run_name": "recalibrated_under_cascade",
        "split": split,
        "imgsz": cfg.imgsz,
        "aug_policy": "n/a",
        "calibration_config": {
            "use_cascade": cfg.use_cascade,
            "use_tta": cfg.use_tta,
            "crop_imgsz": cfg.crop_imgsz,
            "merge_iou": cfg.merge_iou,
        },
        "per_class": per_class,
        "note": ("Ambang disapu dengan kaskade aktif. Berkas evaluasi dari "
                 "pipeline training disapu tanpa kaskade, sehingga ambangnya "
                 "tidak sesuai dengan konfigurasi yang dijalankan di produksi."),
    }

    if output_path:
        Path(output_path).write_text(
            json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Kalibrasi tersimpan: %s", output_path)
    return out


def summarize_changes(result: Dict[str, object]) -> str:
    """Ringkas pergeseran ambang dalam bentuk tabel teks."""
    lines = [f"{'kelas':<12} {'lama':>6} {'baru':>6} {'geser':>7}  dioptimalkan",
             "-" * 56]
    for name, info in result["per_class"].items():
        old = info.get("previous_threshold")
        new = info["recommended_threshold"]
        if old is None:
            lines.append(f"{name:<12} {'-':>6} {new:>6.2f} {'-':>7}  {info['optimized_for']}")
        else:
            lines.append(f"{name:<12} {old:>6.2f} {new:>6.2f} {new - old:>+7.2f}  "
                         f"{info['optimized_for']}")
    return "\n".join(lines)
