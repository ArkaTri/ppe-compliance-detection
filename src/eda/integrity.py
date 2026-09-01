"""
Integrity: pemeriksaan kesehatan dataset SEBELUM analisis statistik.

Alasan modul ini dijalankan pertama: statistik yang dihitung di atas data
kotor menghasilkan kesimpulan yang salah tapi terlihat meyakinkan. Semua
temuan di sini harus dibereskan (atau dijelaskan) sebelum training.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .config import EDAConfig


def _file_hash(path: str, chunk: int = 1 << 16) -> str:
    """MD5 isi file untuk deteksi duplikat byte-identical."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def find_orphan_labels(cfg: EDAConfig, images_df: pd.DataFrame) -> List[str]:
    """File label yang tidak punya pasangan gambar."""
    orphans: List[str] = []
    for split in cfg.available_splits():
        lbl_dir = cfg.labels_dir(split)
        if not lbl_dir.is_dir():
            continue
        known = set(images_df.loc[images_df["split"] == split, "image_id"])
        for lbl in lbl_dir.glob("*.txt"):
            if lbl.stem not in known:
                orphans.append(str(lbl))
    return orphans


def check_coordinates(boxes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Deteksi box dengan koordinat di luar rentang [0,1] atau berdimensi nol.

    Box yang keluar batas biasanya sisa proses crop/resize yang tidak bersih
    dan akan diam-diam di-clip oleh trainer.
    """
    if boxes_df.empty:
        return boxes_df

    out_of_range = (
        (boxes_df[["x1", "y1"]] < -1e-6).any(axis=1)
        | (boxes_df[["x2", "y2"]] > 1 + 1e-6).any(axis=1)
    )
    degenerate = (boxes_df["w"] <= 0) | (boxes_df["h"] <= 0)

    flagged = boxes_df[out_of_range | degenerate].copy()
    flagged["issue"] = ""
    flagged.loc[out_of_range[flagged.index], "issue"] = "out_of_range"
    flagged.loc[degenerate[flagged.index], "issue"] = "degenerate"
    return flagged


def find_duplicates(images_df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Kelompokkan gambar yang identik secara byte.

    Duplikat yang tersebar antara train dan test menyebabkan data leakage:
    skor test terlihat bagus padahal model sudah pernah melihat gambar itu.
    """
    hashes: Dict[str, List[str]] = {}
    for row in images_df.itertuples():
        if not row.readable:
            continue
        digest = _file_hash(row.image_path)
        hashes.setdefault(digest, []).append(f"{row.split}/{row.image_id}")
    return {k: v for k, v in hashes.items() if len(v) > 1}


def run(cfg: EDAConfig, images_df: pd.DataFrame, boxes_df: pd.DataFrame,
        parse_errors: List[str]) -> dict:
    """Jalankan seluruh pemeriksaan integritas dan kembalikan ringkasan."""

    unreadable = images_df.loc[~images_df["readable"], "image_path"].tolist()
    missing_label = images_df.loc[~images_df["has_label_file"], "image_id"].tolist()
    empty_label = images_df.loc[
        images_df["has_label_file"] & (images_df["n_boxes"] == 0), "image_id"
    ].tolist()

    bad_coords = check_coordinates(boxes_df)
    dupes = find_duplicates(images_df)

    # duplikat yang melintasi split = leakage, jauh lebih serius
    cross_split_dupes = {
        k: v for k, v in dupes.items()
        if len({item.split("/")[0] for item in v}) > 1
    }

    invalid_ids = (
        boxes_df.loc[~boxes_df["valid_class_id"], "class_id"].unique().tolist()
        if not boxes_df.empty else []
    )

    return {
        "n_images": int(len(images_df)),
        "n_boxes": int(len(boxes_df)),
        "unreadable_images": unreadable,
        "images_without_label_file": missing_label,
        "images_with_empty_label": empty_label,
        "n_background_images": len(empty_label),
        "parse_errors": parse_errors[:50],
        "n_parse_errors": len(parse_errors),
        "invalid_class_ids": invalid_ids,
        "n_bad_coordinate_boxes": int(len(bad_coords)),
        "bad_coordinate_sample": bad_coords.head(20).to_dict("records"),
        "n_duplicate_groups": len(dupes),
        "cross_split_duplicates": cross_split_dupes,
        "n_cross_split_duplicates": len(cross_split_dupes),
    }
