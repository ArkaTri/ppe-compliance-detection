"""
Audit log: mengubah demo menjadi alat kerja.

Safety officer tidak membutuhkan gambar berkotak. Yang mereka butuhkan
adalah catatan yang dapat diaudit: kapan diperiksa, pekerja mana yang
bermasalah, apa dasar keputusannya, dan berapa tingkat kepatuhan dari
waktu ke waktu.

Kepatuhan APD adalah kewajiban hukum di Indonesia - UU No. 1 Tahun 1970,
PP No. 50 Tahun 2012 tentang SMK3, dan Permenaker No. 8 Tahun 2010 tentang
APD. Ketika terjadi insiden atau inspeksi, yang diminta adalah dokumentasi,
bukan tangkapan layar.

Setiap baris memuat alasan keputusan, bukan hanya vonisnya. Catatan audit
yang tidak dapat dipertanyakan bukanlah catatan audit.
"""

from __future__ import annotations

import csv
import io
from datetime import datetime
from typing import Dict, List, Optional

FIELDS = [
    "timestamp", "image_id", "worker_index", "overall_status",
    "head_status", "head_evidence", "head_confidence", "head_reason",
    "torso_status", "torso_evidence", "torso_confidence", "torso_reason",
    "person_confidence", "bbox", "notes",
]


def records_from_report(report: Dict[str, object],
                        timestamp: Optional[str] = None) -> List[dict]:
    """Ubah satu laporan gambar menjadi baris-baris audit per pekerja."""
    ts = timestamp or datetime.now().isoformat(timespec="seconds")
    image_id = report.get("image_id", "unknown")
    rows: List[dict] = []

    for w in report.get("workers", []):
        head, torso = w["head"], w["torso"]
        rows.append({
            "timestamp": ts,
            "image_id": image_id,
            "worker_index": w["index"],
            "overall_status": w["overall"],
            "head_status": head["status"],
            "head_evidence": head.get("evidence") or "",
            "head_confidence": head.get("confidence"),
            "head_reason": head["reason"],
            "torso_status": torso["status"],
            "torso_evidence": torso.get("evidence") or "",
            "torso_confidence": torso.get("confidence"),
            "torso_reason": torso["reason"],
            "person_confidence": w["person_confidence"],
            "bbox": " ".join(str(v) for v in w["bbox"]),
            "notes": "; ".join(w.get("notes", [])),
        })
    return rows


def to_csv(rows: List[dict]) -> str:
    """Serialisasi baris audit menjadi CSV (string, siap diunduh)."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=FIELDS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def aggregate(reports: List[Dict[str, object]]) -> Dict[str, object]:
    """
    Ringkasan lintas gambar - tampilan tingkat lokasi kerja.

    Tingkat kepatuhan dihitung terhadap pekerja yang DAPAT dinilai.
    Memasukkan yang perlu tinjau ke penyebut akan menghukum sistem karena
    bersikap jujur soal ketidakpastian; angkanya tetap dilaporkan terpisah
    agar beban tinjau manual terlihat.
    """
    n_images = len(reports)
    n_workers = n_ok = n_violation = n_review = 0
    n_orphans = 0
    worst: List[tuple] = []

    for r in reports:
        s = r.get("summary", {})
        n_workers += s.get("n_workers", 0)
        n_ok += s.get("n_compliant", 0)
        n_violation += s.get("n_violation", 0)
        n_review += s.get("n_needs_review", 0)
        n_orphans += len(s.get("orphan_ppe", []))

        if s.get("n_violation", 0) > 0:
            worst.append((s["n_violation"], r.get("image_id", "?")))

    worst.sort(reverse=True)
    assessable = n_ok + n_violation

    return {
        "n_images": n_images,
        "n_workers": n_workers,
        "n_compliant": n_ok,
        "n_violation": n_violation,
        "n_needs_review": n_review,
        "assessable": assessable,
        "compliance_rate": round(n_ok / assessable, 4) if assessable else None,
        "review_load": round(n_review / n_workers, 4) if n_workers else None,
        "n_orphan_ppe": n_orphans,
        "images_with_violations": worst[:10],
    }


def summary_text(agg: Dict[str, object]) -> str:
    """Ringkasan satu paragraf, untuk ditempel ke email atau laporan."""
    if not agg.get("n_workers"):
        return "Tidak ada pekerja terdeteksi pada gambar yang diperiksa."

    rate = agg.get("compliance_rate")
    rate_txt = f"{rate:.0%}" if rate is not None else "tidak dapat dihitung"

    parts = [
        f"Diperiksa {agg['n_images']} gambar, {agg['n_workers']} pekerja terdeteksi.",
        f"Tingkat kepatuhan {rate_txt} dari {agg['assessable']} pekerja yang dapat dinilai.",
    ]
    if agg["n_violation"]:
        parts.append(f"{agg['n_violation']} pelanggaran APD tercatat.")
    if agg["n_needs_review"]:
        parts.append(f"{agg['n_needs_review']} pekerja perlu ditinjau manual "
                     "karena APD tidak dapat dipastikan.")
    if agg["n_orphan_ppe"]:
        parts.append(f"{agg['n_orphan_ppe']} APD terdeteksi tanpa pemilik - "
                     "kemungkinan ada pekerja yang tidak terdeteksi.")
    return " ".join(parts)
