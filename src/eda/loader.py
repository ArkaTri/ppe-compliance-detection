"""
Loader: mengubah struktur folder YOLO menjadi dua DataFrame yang rapi.

Prinsip desain: SEMUA modul analisis di hilir hanya bekerja di atas DataFrame.
Tidak ada modul lain yang boleh menyentuh filesystem. Ini membuat setiap
analisis bisa diuji tanpa dataset asli, dan menghindari pembacaan disk berulang.

Output:
    images_df : satu baris per gambar  (metadata file + dimensi)
    boxes_df  : satu baris per bounding box (koordinat ternormalisasi + absolut)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from PIL import Image

from .config import EDAConfig

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _read_image_size(path: Path) -> Tuple[int, int, bool]:
    """Baca dimensi gambar tanpa memuat seluruh pixel. Return (w, h, ok)."""
    try:
        with Image.open(path) as im:
            im.verify()  # deteksi file korup
        with Image.open(path) as im:
            return im.width, im.height, True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Gambar tidak terbaca: %s (%s)", path.name, exc)
        return 0, 0, False


def _parse_label_file(path: Path) -> Tuple[List[tuple], List[str]]:
    """
    Parse satu file label YOLO.

    Return (rows, errors) dengan rows = [(class_id, xc, yc, w, h), ...].
    Baris yang cacat dilewati dan dicatat, bukan membuat proses berhenti.
    """
    rows: List[tuple] = []
    errors: List[str] = []

    if not path.exists():
        return rows, errors

    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            errors.append(f"{path.name}:{lineno} jumlah kolom < 5")
            continue
        try:
            cid = int(float(parts[0]))
            xc, yc, w, h = (float(v) for v in parts[1:5])
        except ValueError:
            errors.append(f"{path.name}:{lineno} nilai non-numerik")
            continue
        rows.append((cid, xc, yc, w, h))

    return rows, errors


def load_dataset(cfg: EDAConfig) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Pindai seluruh split dan bangun DataFrame gambar + bounding box."""

    image_records: List[dict] = []
    box_records: List[dict] = []
    parse_errors: List[str] = []

    n_classes = len(cfg.class_names)

    for split in cfg.available_splits():
        img_dir = cfg.images_dir(split)
        lbl_dir = cfg.labels_dir(split)

        image_paths = sorted(
            p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
        )
        logger.info("Split '%s': %d gambar ditemukan", split, len(image_paths))

        for img_path in image_paths:
            width, height, readable = _read_image_size(img_path)
            lbl_path = lbl_dir / f"{img_path.stem}.txt"

            rows, errs = _parse_label_file(lbl_path)
            parse_errors.extend(errs)

            image_records.append(
                {
                    "split": split,
                    "image_id": img_path.stem,
                    "image_path": str(img_path),
                    "label_path": str(lbl_path),
                    "width": width,
                    "height": height,
                    "readable": readable,
                    "has_label_file": lbl_path.exists(),
                    "n_boxes": len(rows),
                }
            )

            for idx, (cid, xc, yc, bw, bh) in enumerate(rows):
                valid_id = 0 <= cid < n_classes
                box_records.append(
                    {
                        "split": split,
                        "image_id": img_path.stem,
                        "box_idx": idx,
                        "class_id": cid,
                        "class_name": cfg.class_names[cid] if valid_id else f"UNKNOWN_{cid}",
                        "valid_class_id": valid_id,
                        "xc": xc,
                        "yc": yc,
                        "w": bw,
                        "h": bh,
                        # koordinat sudut ternormalisasi (0-1)
                        "x1": xc - bw / 2,
                        "y1": yc - bh / 2,
                        "x2": xc + bw / 2,
                        "y2": yc + bh / 2,
                        "area_frac": bw * bh,  # fraksi luas gambar
                        "img_w": width,
                        "img_h": height,
                    }
                )

    images_df = pd.DataFrame(image_records)
    boxes_df = pd.DataFrame(box_records)

    if not boxes_df.empty:
        # koordinat absolut (pixel) berguna untuk analisis ukuran objek nyata
        boxes_df["w_px"] = boxes_df["w"] * boxes_df["img_w"]
        boxes_df["h_px"] = boxes_df["img_h"] * boxes_df["h"]
        boxes_df["area_px"] = boxes_df["w_px"] * boxes_df["h_px"]
        boxes_df["aspect_ratio"] = boxes_df["w_px"] / boxes_df["h_px"].replace(0, pd.NA)

    logger.info(
        "Total: %d gambar, %d bounding box, %d error parsing",
        len(images_df), len(boxes_df), len(parse_errors),
    )
    return images_df, boxes_df, parse_errors
