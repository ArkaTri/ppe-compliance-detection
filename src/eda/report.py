"""
Report: menggabungkan seluruh temuan menjadi artefak yang bisa diserahkan.

Menghasilkan dua berkas:
  eda_report.json  - mesin-terbaca, dipakai modul training di hilir
  eda_report.md    - manusia-terbaca, langsung bisa masuk README GitHub

Bagian terpenting adalah 'Keputusan Teknis': tiap keputusan training
ditautkan ke temuan spesifik, bukan ke preferensi pribadi.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from .config import EDAConfig


def _json_safe(obj: Any) -> Any:
    """Bersihkan objek agar bisa diserialisasi (buang DataFrame internal)."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items() if not k.startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        return obj.to_dict()
    if hasattr(obj, "item"):  # numpy scalar
        try:
            return obj.item()
        except Exception:  # noqa: BLE001
            return str(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _decisions(dist: dict, geom: dict, assoc: dict) -> list:
    """Turunkan keputusan training dari temuan EDA."""
    out = []

    viol = dist.get("violations", {})
    if viol:
        if viol.get("sufficient"):
            out.append(
                f"**Bobot kelas**: total {viol['total_violation_instances']} instance "
                "pelanggaran memadai untuk dilatih langsung. Tetap pantau recall "
                "per kelas, bukan hanya mAP agregat."
            )
        else:
            kurang = ", ".join(f"`{c}` ({n})" for c, n in
                               viol.get("insufficient_classes", {}).items())
            out.append(
                f"**Bobot kelas**: {kurang} berada di bawah ambang "
                f"{viol['threshold']} instance. Jangan andalkan detector kelas ini "
                "sebagai satu-satunya sumber pelanggaran; gunakan juga inferensi "
                "dari ketiadaan APD pada person yang terdeteksi. Terapkan "
                "oversampling gambar yang memuat kelas tersebut dan confidence "
                "threshold yang lebih rendah khusus untuknya saat inferensi."
            )

    imb = dist.get("imbalance", {})
    if imb.get("imbalance_ratio", 0) > 10:
        out.append(
            f"**Ketidakseimbangan**: rasio {imb['imbalance_ratio']:.1f}x antara "
            f"'{imb['max_class']}' dan '{imb['min_class']}'. mAP agregat akan "
            "didominasi kelas mayoritas; laporkan metrik per kelas secara eksplisit."
        )

    rec = geom.get("imgsz_recommendation", {})
    if rec:
        out.append(
            f"**imgsz**: {rec['recommended_imgsz']} - {rec['reason']} "
            f"({rec['ppe_small_object_share']:.0%} box APD tergolong objek kecil)."
        )

    res = geom.get("image_resolution", {})
    if res and not res.get("is_uniform", True):
        out.append(
            f"**Resize**: dataset memuat {res['n_unique_resolutions']} resolusi berbeda. "
            "Gunakan letterbox (padding, mempertahankan aspect ratio), bukan stretch, "
            "agar geometri objek tidak terdistorsi."
        )

    if assoc.get("feasible"):
        out.append(
            f"**Association engine**: LAYAK. attach rate "
            f"{assoc['attach_rate']:.1%} pada ambang containment "
            f"{assoc['containment_threshold']}. Lanjutkan desain per-worker "
            "compliance."
        )
    else:
        out.append(f"**Association engine**: {assoc.get('verdict', 'perlu ditinjau')}")

    if assoc.get("ambiguity_rate", 0) > 0.15:
        out.append(
            f"**Resolusi ambiguitas**: {assoc['ambiguity_rate']:.1%} box APD dapat "
            "diklaim lebih dari satu person. Wajib gunakan penugasan berbasis skor "
            "(greedy / Hungarian), bukan pencocokan pertama-yang-ditemukan."
        )

    out.append(
        "**Augmentasi**: matikan seluruh augmentasi geometris secara eksplisit "
        "(`fliplr=0.0, flipud=0.0, mosaic=0.0, degrees=0.0, translate=0.0, "
        "scale=0.0, shear=0.0, perspective=0.0`) sesuai panduan capstone. "
        "Default Ultralytics MENGAKTIFKAN fliplr dan mosaic; tanpa penonaktifan "
        "eksplisit panduan tersebut dilanggar tanpa disadari."
    )
    return out


def build_markdown(cfg: EDAConfig, integ: dict, dist: dict, geom: dict,
                   assoc: dict, figures: dict) -> str:
    """
    Rangkai semua hasil analisis (integrity, distribution, geometry,
    association) menjadi satu laporan Markdown siap dibaca manusia.
    """
    L = []
    A = L.append

    A("# Laporan EDA - Construction Safety PPE Dataset")
    A("")
    A(f"_Dihasilkan otomatis pada {datetime.now():%Y-%m-%d %H:%M}_")
    A("")
    A(f"- Sumber: `{cfg.dataset_root}`")
    A(f"- Kelas: {', '.join(cfg.class_names)}")
    A(f"- Total gambar: **{integ['n_images']}** | Total bounding box: **{integ['n_boxes']}**")
    A("")

    # -- integritas -------------------------------------------------------
    A("## 1. Integritas Data")
    A("")
    A("| Pemeriksaan | Hasil |")
    A("|---|---|")
    A(f"| Gambar tidak terbaca | {len(integ['unreadable_images'])} |")
    A(f"| Gambar tanpa file label | {len(integ['images_without_label_file'])} |")
    A(f"| Gambar berlabel kosong (background) | {integ['n_background_images']} |")
    A(f"| Baris label cacat | {integ['n_parse_errors']} |")
    A(f"| Box dengan koordinat bermasalah | {integ['n_bad_coordinate_boxes']} |")
    A(f"| Grup gambar duplikat | {integ['n_duplicate_groups']} |")
    A(f"| **Duplikat lintas split (leakage)** | **{integ['n_cross_split_duplicates']}** |")
    A("")
    if integ["n_cross_split_duplicates"] > 0:
        A("> ⚠️ Duplikat lintas split terdeteksi. Gambar yang sama muncul di lebih "
          "dari satu split, sehingga skor evaluasi menjadi optimistis palsu. "
          "Hapus duplikat sebelum training dan catat tindakan ini di laporan.")
        A("")

    # -- distribusi -------------------------------------------------------
    A("## 2. Sebaran Kelas")
    A("")
    counts = dist.get("_counts_df")
    if counts is not None and not counts.empty:
        A(counts.to_markdown())
        A("")
    imb = dist.get("imbalance", {})
    if imb:
        A(f"Rasio ketidakseimbangan: **{imb['imbalance_ratio']:.1f}x** "
          f"(`{imb['max_class']}` = {imb['max_count']} vs "
          f"`{imb['min_class']}` = {imb['min_count']}).")
        A("")
    viol = dist.get("violations", {})
    if viol:
        A(f"Instance kelas pelanggaran: **{viol['total_violation_instances']}** "
          f"({viol['per_class']}).")
        A("")
        kurang = viol.get("insufficient_classes", {})
        if kurang:
            detail = ", ".join(f"`{c}` = {n}" for c, n in kurang.items())
            A(f"> ⚠️ **TERBATAS per kelas**: {detail} berada di bawah ambang "
              f"{viol['threshold']} instance. Ambang diperiksa PER KELAS, bukan "
              "agregat: total yang besar dapat menyembunyikan kelas kritis yang "
              "langka.")
        else:
            A(f"Status: **MEMADAI** - seluruh kelas pelanggaran melampaui ambang "
              f"{viol['threshold']} instance.")
        A("")
    if figures.get("class_distribution"):
        A("![Sebaran kelas](figures/class_distribution.png)")
        A("")

    # -- geometri ---------------------------------------------------------
    A("## 3. Geometri Objek")
    A("")
    per_class = geom.get("_per_class_df")
    if per_class is not None and not per_class.empty:
        A(per_class.to_markdown())
        A("")
    rec = geom.get("imgsz_recommendation", {})
    if rec:
        A(f"Rekomendasi `imgsz`: **{rec['recommended_imgsz']}** - {rec['reason']}.")
        A("")
    if figures.get("size_distribution"):
        A("![Ukuran objek](figures/size_distribution.png)")
        A("")

    # -- asosiasi ---------------------------------------------------------
    A("## 4. Uji Kelayakan Asosiasi Person - APD")
    A("")
    A("> Bagian ini menentukan apakah arsitektur *person-centric compliance* "
      "dapat dibangun. Metrik utama menggunakan **containment ratio** "
      "(`area irisan / area box APD`), bukan IoU, karena box APD jauh lebih "
      "kecil daripada box tubuh sehingga IoU standar menyesatkan.")
    A("")
    if assoc.get("feasible") is False and "reason" in assoc:
        A(f"**Hasil: TIDAK LAYAK** - {assoc['reason']}")
        A("")
    else:
        A("| Metrik | Nilai |")
        A("|---|---|")
        A(f"| Ambang containment | {assoc['containment_threshold']} |")
        A(f"| Gambar memuat APD | {assoc['n_images_with_ppe']} |")
        A(f"| Person coverage | {assoc['person_coverage']:.1%} |")
        A(f"| Attach rate | {assoc['attach_rate']:.1%} |")
        A(f"| Ambiguity rate | {assoc['ambiguity_rate']:.1%} |")
        A("")
        A(f"**Verdict:** {assoc['verdict']}")
        A("")
        if assoc.get("attach_rate_per_class"):
            A("Attach rate per kelas APD:")
            A("")
            for k, v in assoc["attach_rate_per_class"].items():
                A(f"- `{k}`: {v:.1%}")
            A("")
        if figures.get("association_sensitivity"):
            A("![Sensitivitas asosiasi](figures/association_sensitivity.png)")
            A("")
        zv = assoc.get("zone_validation", {})
        if zv:
            A("### Validasi asumsi zona anatomis")
            A("")
            A("| Kelas | n | p05 | median | p95 | Zona harapan | Di dalam zona |")
            A("|---|---|---|---|---|---|---|")
            for cls, s in zv.items():
                A(f"| {cls} | {s['n']} | {s['rel_y_p05']} | {s['rel_y_p50']} | "
                  f"{s['rel_y_p95']} | {s['expected_zone']} | "
                  f"{s['within_expected_zone']:.1%} |")
            A("")
            A("Jika kolom terakhir bernilai rendah, constraint geometris hanya "
              "boleh dipakai sebagai penambah skor, bukan filter keras.")
            A("")

    # -- keputusan --------------------------------------------------------
    A("## 5. Keputusan Teknis yang Diturunkan dari EDA")
    A("")
    for i, d in enumerate(_decisions(dist, geom, assoc), 1):
        A(f"{i}. {d}")
    A("")

    A("## 6. Atribusi Dataset")
    A("")
    A("Dataset: *construction safety* (Roboflow Universe, workspace "
      "`personal-project-kej16`), bagian dari benchmark RF100. "
      "Lisensi: **CC BY 4.0**.")
    A("")

    return "\n".join(L)


def run(cfg: EDAConfig, integ: dict, dist: dict, geom: dict,
        assoc: dict, figures: dict) -> Dict[str, str]:
    """
    Titik masuk modul report: tulis hasil EDA ke dua format sekaligus —
    eda_report.json (mesin-terbaca) dan eda_report.md (manusia-terbaca).
    """
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_root": str(cfg.dataset_root),
        "class_names": cfg.class_names,
        "integrity": _json_safe(integ),
        "distribution": _json_safe(dist),
        "geometry": _json_safe(geom),
        "association": _json_safe(assoc),
        "decisions": _decisions(dist, geom, assoc),
    }

    json_path = cfg.output_dir / "eda_report.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                         encoding="utf-8")

    md_path = cfg.output_dir / "eda_report.md"
    md_path.write_text(build_markdown(cfg, integ, dist, geom, assoc, figures),
                       encoding="utf-8")

    return {"json": str(json_path), "markdown": str(md_path)}
