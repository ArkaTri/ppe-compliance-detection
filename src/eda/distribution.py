"""
Distribution: sebaran kelas, ketidakseimbangan, dan ko-okurensi.

Pertanyaan kunci yang dijawab modul ini:
  1. Berapa langka kelas pelanggaran (no-helmet, no-vest)?
  2. Apakah komposisi kelas konsisten antar split? (split yang tidak
     terstratifikasi membuat skor validasi tidak bisa dipercaya)
  3. Kelas apa yang cenderung muncul bersamaan? (dasar penalaran association)
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .config import EDAConfig, VIOLATION_CLASSES


def class_counts(boxes_df: pd.DataFrame) -> pd.DataFrame:
    """Matriks jumlah instance: baris = kelas, kolom = split (+ total)."""
    if boxes_df.empty:
        return pd.DataFrame()
    tab = pd.crosstab(boxes_df["class_name"], boxes_df["split"])
    tab["TOTAL"] = tab.sum(axis=1)
    return tab.sort_values("TOTAL", ascending=False)


def imbalance_report(counts: pd.DataFrame) -> Dict[str, float]:
    """
    Ukuran ketidakseimbangan.

    imbalance_ratio = instance kelas terbanyak / kelas tersedikit.
    Nilai > 10 biasanya sudah cukup untuk membuat model mengabaikan
    kelas minoritas walau mAP keseluruhan terlihat wajar.
    """
    if counts.empty:
        return {}
    total = counts["TOTAL"]
    return {
        "max_class": str(total.idxmax()),
        "min_class": str(total.idxmin()),
        "max_count": int(total.max()),
        "min_count": int(total.min()),
        "imbalance_ratio": float(total.max() / max(total.min(), 1)),
    }


def split_proportion_drift(counts: pd.DataFrame) -> pd.DataFrame:
    """
    Bandingkan proporsi tiap kelas antar split.

    Jika proporsi kelas pelanggaran di test jauh berbeda dari train,
    metrik test tidak mengukur hal yang sama dengan yang dilatih.
    """
    if counts.empty:
        return pd.DataFrame()
    split_cols = [c for c in counts.columns if c != "TOTAL"]
    props = counts[split_cols] / counts[split_cols].sum(axis=0)
    props["max_abs_drift"] = props.max(axis=1) - props.min(axis=1)
    return props.round(4)


def per_image_stats(images_df: pd.DataFrame, boxes_df: pd.DataFrame) -> Dict[str, float]:
    """Statistik kepadatan objek per gambar (memengaruhi risiko oklusi)."""
    if boxes_df.empty:
        return {}
    per_img = boxes_df.groupby(["split", "image_id"]).size()
    return {
        "mean_boxes_per_image": float(per_img.mean()),
        "median_boxes_per_image": float(per_img.median()),
        "p95_boxes_per_image": float(np.percentile(per_img, 95)),
        "max_boxes_per_image": int(per_img.max()),
    }


def cooccurrence_matrix(boxes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Berapa banyak gambar yang memuat pasangan kelas (A, B) sekaligus.

    Diagonal = jumlah gambar yang memuat kelas tersebut.
    Baris 'person' pada matriks ini adalah petunjuk awal apakah anotator
    benar-benar melabeli person bersama APD-nya.
    """
    if boxes_df.empty:
        return pd.DataFrame()
    presence = (
        boxes_df.assign(one=1)
        .pivot_table(index=["split", "image_id"], columns="class_name",
                     values="one", aggfunc="max", fill_value=0)
    )
    return presence.T.dot(presence)


def violation_summary(counts: pd.DataFrame, cfg: EDAConfig) -> Dict[str, object]:
    """Ringkasan khusus kelas pelanggaran + verdict terhadap ambang minimum."""
    if counts.empty:
        return {}
    present = [c for c in VIOLATION_CLASSES if c in counts.index]
    per_class = {c: int(counts.loc[c, "TOTAL"]) for c in present}
    total = sum(per_class.values())
    insufficient = {c: n for c, n in per_class.items()
                    if n < cfg.min_violation_instances}
    return {
        "per_class": per_class,
        "total_violation_instances": total,
        "threshold": cfg.min_violation_instances,
        "insufficient_classes": insufficient,
        "sufficient": len(insufficient) == 0,
    }

def run(cfg: EDAConfig, images_df: pd.DataFrame, boxes_df: pd.DataFrame) -> dict:
    """
    Titik masuk modul distribution: jalankan seluruh analisis sebaran kelas
    (count, imbalance, drift antar split, ko-okurensi, ringkasan pelanggaran).
    """
    counts = class_counts(boxes_df)
    return {
        "class_counts": counts.to_dict() if not counts.empty else {},
        "imbalance": imbalance_report(counts),
        "split_proportions": split_proportion_drift(counts).to_dict(),
        "per_image": per_image_stats(images_df, boxes_df),
        "cooccurrence": cooccurrence_matrix(boxes_df).to_dict(),
        "violations": violation_summary(counts, cfg),
        "_counts_df": counts,  # dipakai modul visualize
    }
