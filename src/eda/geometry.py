"""
Geometry: analisis ukuran dan bentuk bounding box.

Keputusan yang bergantung pada modul ini:
  - Nilai `imgsz` saat training. Jika mayoritas box APD masuk kategori
    'small', menaikkan imgsz dari 640 ke 960 biasanya lebih berdampak
    daripada menambah epoch.
  - Ekspektasi realistis terhadap mAP. Objek kecil secara inheren punya
    mAP lebih rendah; ini konteks yang harus ada di laporan agar angka
    tidak dibaca sebagai kegagalan model.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .config import EDAConfig, PPE_CLASSES


def size_bucket(area_frac: float, cfg: EDAConfig) -> str:
    if area_frac < cfg.small_area_frac:
        return "small"
    if area_frac < cfg.medium_area_frac:
        return "medium"
    return "large"


def per_class_geometry(boxes_df: pd.DataFrame) -> pd.DataFrame:
    """Statistik ukuran per kelas dalam fraksi luas gambar dan pixel."""
    if boxes_df.empty:
        return pd.DataFrame()
    g = boxes_df.groupby("class_name")
    stats = pd.DataFrame({
        "n": g.size(),
        "area_frac_mean": g["area_frac"].mean(),
        "area_frac_p50": g["area_frac"].median(),
        "area_frac_p95": g["area_frac"].quantile(0.95),
        "w_px_p50": g["w_px"].median(),
        "h_px_p50": g["h_px"].median(),
        "aspect_p50": g["aspect_ratio"].median(),
    })
    # sisi ekuivalen: akar dari luas, memudahkan intuisi "sekecil apa"
    stats["equiv_side_px_p50"] = np.sqrt(g["area_px"].median())
    return stats.round(4).sort_values("area_frac_p50")


def size_distribution(boxes_df: pd.DataFrame, cfg: EDAConfig) -> pd.DataFrame:
    """Proporsi small/medium/large per kelas."""
    if boxes_df.empty:
        return pd.DataFrame()
    tmp = boxes_df.copy()
    tmp["bucket"] = tmp["area_frac"].apply(lambda a: size_bucket(a, cfg))
    tab = pd.crosstab(tmp["class_name"], tmp["bucket"], normalize="index")
    return tab.round(4)


def image_resolution_profile(images_df: pd.DataFrame) -> Dict[str, object]:
    """
    Profil resolusi asli gambar.

    Relevan karena dataset ini (berbeda dari dataset Calory) TIDAK melalui
    pre-processing resize di Roboflow, sehingga resolusi bisa beragam dan
    keputusan resize sepenuhnya ada di tangan kita.
    """
    valid = images_df[images_df["readable"]]
    if valid.empty:
        return {}
    combos = valid.groupby(["width", "height"]).size().sort_values(ascending=False)
    return {
        "n_unique_resolutions": int(len(combos)),
        "top_resolutions": {f"{w}x{h}": int(n) for (w, h), n in combos.head(8).items()},
        "width_median": float(valid["width"].median()),
        "height_median": float(valid["height"].median()),
        "is_uniform": bool(len(combos) == 1),
    }


def imgsz_recommendation(size_dist: pd.DataFrame) -> Dict[str, object]:
    """
    Rekomendasi imgsz berbasis proporsi objek kecil pada kelas APD.

    Ini heuristik, bukan kebenaran mutlak; angkanya tetap harus diverifikasi
    lewat eksperimen (lihat ablation study di laporan).
    """
    if size_dist.empty or "small" not in size_dist.columns:
        return {}
    ppe_rows = [c for c in PPE_CLASSES if c in size_dist.index]
    if not ppe_rows:
        return {}
    small_share = float(size_dist.loc[ppe_rows, "small"].mean())

    if small_share >= 0.50:
        rec, reason = 960, "mayoritas box APD tergolong objek kecil"
    elif small_share >= 0.25:
        rec, reason = 768, "porsi objek kecil cukup signifikan"
    else:
        rec, reason = 640, "ukuran objek APD relatif memadai pada 640"

    return {
        "ppe_small_object_share": round(small_share, 4),
        "recommended_imgsz": rec,
        "reason": reason,
        "note": "Verifikasi dengan membandingkan mAP@50-95 pada dua nilai imgsz.",
    }


def run(cfg: EDAConfig, images_df: pd.DataFrame, boxes_df: pd.DataFrame) -> dict:
    per_class = per_class_geometry(boxes_df)
    size_dist = size_distribution(boxes_df, cfg)
    return {
        "per_class_geometry": per_class.to_dict(),
        "size_distribution": size_dist.to_dict(),
        "image_resolution": image_resolution_profile(images_df),
        "imgsz_recommendation": imgsz_recommendation(size_dist),
        "_per_class_df": per_class,
        "_size_dist_df": size_dist,
    }
