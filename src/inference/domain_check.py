"""
Deteksi ketidaksesuaian domain - mengenali saat model berada di luar
wilayah yang pernah dilatihkan.

Kegagalan yang diserang, teramati langsung pada aplikasi produksi:

    Sebuah foto pekerja migas berseragam coverall oranye reflektif dinilai
    sistem sebagai 5 dari 5 pekerja MELANGGAR. Mereka sebenarnya memakai
    APD yang benar - hanya jenis yang tidak pernah muncul di data latih,
    yang seluruhnya berisi rompi hi-vis potongan terbuka.

    Model tidak salah karena lemah. Model salah karena diberi gambar di luar
    domainnya, dan tidak punya cara menyatakan hal itu.

Tanda tangan yang dipakai:

    Pada foto tersebut, seluruh box `no-vest` muncul pada keyakinan 0.48-0.58 -
    rendah - sementara SELURUH pekerja ditandai melanggar. Kombinasi itu
    berbeda dari lokasi kerja yang benar-benar lalai, di mana pelanggaran
    nyata biasanya terdeteksi dengan keyakinan tinggi karena ketiadaan APD
    memang jelas terlihat.

    Vonis massal + keyakinan rendah = kemungkinan besar model sedang menebak
    pada distribusi yang asing, bukan menemukan pelanggaran sungguhan.

Sikap modul ini konsisten dengan seluruh sistem: bila sebuah angka
berpotensi menyesatkan, angka itu tidak disajikan begitu saja - ia disertai
peringatan tentang mengapa ia mungkin salah. Modul ini TIDAK mengubah vonis
per pekerja; ia hanya menambahkan konteks agar pengguna tidak menindak
laporan yang kemungkinan besar keliru.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Dict, List, Optional

from .config import (
    COMPLIANT,
    NEEDS_REVIEW,
    VIOLATION,
    VIOLATION_INFERRED,
)


@dataclass
class DomainThresholds:
    """
    Ambang untuk memicu peringatan.

    Nilai awal diturunkan dari kasus coverall yang teramati: 100% pekerja
    ditandai melanggar dengan keyakinan bukti rata-rata sekitar 0.53. Ambang
    dipasang sedikit longgar agar kasus serupa tertangkap tanpa membanjiri
    pengguna dengan peringatan pada gambar biasa.
    """
    mass_violation_rate: float = 0.75   # fraksi pekerja yang dinilai melanggar
    low_evidence_conf: float = 0.65     # rata-rata keyakinan bukti pelanggaran
    min_workers: int = 3                # di bawah ini, sampelnya terlalu kecil
    low_overall_conf: float = 0.55      # rata-rata keyakinan seluruh deteksi APD


def _violation_evidence_confidences(report: Dict[str, object]) -> List[float]:
    """Kumpulkan keyakinan bukti pada bagian tubuh yang divonis melanggar."""
    out: List[float] = []
    for w in report.get("workers", []):
        for part in ("head", "torso"):
            info = w[part]
            if info["status"] in (VIOLATION, VIOLATION_INFERRED):
                c = info.get("confidence")
                if c is not None:
                    out.append(float(c))
    return out


def _all_ppe_confidences(report: Dict[str, object]) -> List[float]:
    """Keyakinan seluruh box APD yang terasosiasi, apa pun vonisnya."""
    out: List[float] = []
    for w in report.get("_workers", []):
        for det in (w.head, w.torso):
            if det is not None:
                out.append(float(det.confidence))
    return out


def assess_domain(report: Dict[str, object],
                  thresholds: Optional[DomainThresholds] = None) -> Dict[str, object]:
    """
    Nilai apakah gambar ini kemungkinan berada di luar domain model.

    Return dict berisi indikator dan, bila terpicu, penjelasan yang dapat
    ditampilkan ke pengguna.
    """
    t = thresholds or DomainThresholds()
    s = report.get("summary", {})

    n_workers = s.get("n_workers", 0)
    n_violation = s.get("n_violation", 0)
    assessable = s.get("assessable", 0)

    evid = _violation_evidence_confidences(report)
    allc = _all_ppe_confidences(report)

    violation_rate = (n_violation / assessable) if assessable else 0.0
    mean_evid = round(mean(evid), 3) if evid else None
    mean_all = round(mean(allc), 3) if allc else None

    reasons: List[str] = []

    # Sinyal 1: vonis massal dengan bukti berkeyakinan rendah.
    if (n_workers >= t.min_workers
            and violation_rate >= t.mass_violation_rate
            and mean_evid is not None
            and mean_evid <= t.low_evidence_conf):
        reasons.append(
            f"{n_violation} dari {assessable} pekerja dinilai melanggar, "
            f"tetapi keyakinan rata-rata buktinya hanya {mean_evid:.2f}. "
            "Pelanggaran yang nyata biasanya terdeteksi dengan keyakinan "
            "lebih tinggi karena ketiadaan APD terlihat jelas."
        )

    # Sinyal 2: seluruh deteksi APD berkeyakinan rendah, terlepas dari vonis.
    if (mean_all is not None and len(allc) >= 3
            and mean_all <= t.low_overall_conf):
        reasons.append(
            f"Keyakinan rata-rata seluruh deteksi APD hanya {mean_all:.2f}. "
            "Model tampak ragu pada gambar ini secara keseluruhan."
        )

    flagged = bool(reasons)

    out: Dict[str, object] = {
        "out_of_domain_suspected": flagged,
        "violation_rate": round(violation_rate, 3),
        "mean_violation_evidence_confidence": mean_evid,
        "mean_ppe_confidence": mean_all,
        "n_ppe_detections": len(allc),
        "signals": reasons,
    }

    if flagged:
        out["warning"] = (
            "Kemungkinan gambar ini berada di luar domain model. "
            + " ".join(reasons) +
            " Model dilatih pada rompi hi-vis potongan terbuka; jenis APD "
            "lain - misalnya coverall satu potong - dapat dinilai keliru "
            "sebagai pelanggaran. Periksa hasil ini secara manual sebelum "
            "ditindaklanjuti."
        )

    return out


def summary_line(domain: Dict[str, object]) -> str:
    """Satu baris ringkas untuk dicantumkan pada catatan audit."""
    if not domain.get("out_of_domain_suspected"):
        return ""
    return (f"DOMAIN: diragukan (vonis melanggar "
            f"{domain['violation_rate']:.0%}, keyakinan bukti "
            f"{domain.get('mean_violation_evidence_confidence')})")
