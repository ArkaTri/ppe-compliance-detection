"""
Penilaian kepatuhan per pekerja - logika pelanggaran DUA JALUR.

Mengapa dua jalur, dengan bukti empiris dari eksperimen sendiri:

    Recall detector `no-helmet` pada test set hanya 0.542, dengan
    interval kepercayaan 95% [0.263, 0.793]. Artinya hampir separuh
    pekerja tanpa helm tidak terdeteksi, bahkan setelah threshold
    diturunkan ke 0.2. Penyebabnya jumlah data: hanya 129 instance
    di seluruh dataset, 94 di antaranya di split train.

    Sebaliknya, detector `person` mencapai recall 0.846 dan `helmet`
    0.821 - keduanya dilatih dari ribuan instance.

Karena itu sistem TIDAK bergantung pada detector pelanggaran saja:

    Jalur A - deteksi eksplisit.
        Box `no-helmet` atau `no-vest` terasosiasi ke pekerja.
        Presisi tinggi, recall rendah.

    Jalur B - inferensi dari ketiadaan.
        Ada box `person`, tetapi tidak ada APD apa pun yang terasosiasi
        pada bagian tubuh tersebut. Memanfaatkan dua detector terkuat
        untuk menyimpulkan sesuatu tentang kelas terlemah.

Prinsip fail-safe: bila ketiadaan APD dapat dijelaskan oleh oklusi atau
terpotongnya frame, sistem TIDAK menyimpulkan pelanggaran maupun kepatuhan.
Ia mengembalikan NEEDS_REVIEW. Sistem keselamatan harus gagal secara
terbuka, bukan diam-diam - kesalahan yang paling berbahaya adalah
menyatakan aman sesuatu yang sebenarnya tidak diketahui.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

from .association import Detection, Worker, containment
from .config import (
    COMPLIANT,
    HEAD_NEGATIVE,
    HEAD_POSITIVE,
    InferenceConfig,
    NEEDS_REVIEW,
    TORSO_NEGATIVE,
    TORSO_POSITIVE,
    VIOLATION,
    VIOLATION_INFERRED,
)


@dataclass
class PartStatus:
    status: str
    reason: str
    evidence: Optional[str] = None
    confidence: Optional[float] = None


@dataclass
class WorkerCompliance:
    index: int
    bbox: Tuple[float, float, float, float]
    person_confidence: float
    head: PartStatus
    torso: PartStatus
    overall: str
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["bbox"] = [round(v, 1) for v in self.bbox]
        return d


# --------------------------------------------------------------------------
# Pemeriksaan visibilitas
# --------------------------------------------------------------------------

def _zone_box(person: Detection, zone: Tuple[float, float]) -> Tuple[float, float, float, float]:
    """Kotak wilayah tubuh (kepala / torso) di dalam bbox person."""
    x1, y1, x2, y2 = person.xyxy
    h = y2 - y1
    return (x1, y1 + zone[0] * h, x2, y1 + zone[1] * h)


def _touches_edge(box: Tuple[float, float, float, float],
                  img_w: int, img_h: int, margin: float) -> bool:
    x1, y1, x2, y2 = box
    mx, my = margin * img_w, margin * img_h
    return x1 <= mx or y1 <= my or x2 >= img_w - mx or y2 >= img_h - my


def _occluded_by_other(zone_box: Tuple[float, float, float, float],
                       self_person: Detection,
                       all_persons: List[Detection],
                       thr: float) -> bool:
    """Apakah wilayah tubuh ini tertutup bbox pekerja lain secara signifikan."""
    zx1, zy1, zx2, zy2 = zone_box
    zone_area = max(0.0, zx2 - zx1) * max(0.0, zy2 - zy1)
    if zone_area <= 0:
        return False

    for other in all_persons:
        if other is self_person:
            continue
        ox1, oy1, ox2, oy2 = other.xyxy
        iw = min(zx2, ox2) - max(zx1, ox1)
        ih = min(zy2, oy2) - max(zy1, oy1)
        if iw > 0 and ih > 0 and (iw * ih) / zone_area >= thr:
            return True
    return False


def _assess_absent(part: str, worker: Worker, all_persons: List[Detection],
                   img_w: int, img_h: int,
                   cfg: InferenceConfig) -> PartStatus:
    """
    Tidak ada APD terasosiasi. Tentukan: pelanggaran tersimpulkan, atau
    memang tidak dapat dinilai?
    """
    zone = cfg.head_zone if part == "head" else cfg.torso_zone
    zbox = _zone_box(worker.person, zone)

    if _touches_edge(zbox, img_w, img_h, cfg.edge_margin):
        return PartStatus(
            NEEDS_REVIEW,
            f"wilayah {part} terpotong tepi gambar - tidak dapat dinilai",
        )

    if _occluded_by_other(zbox, worker.person, all_persons, cfg.occlusion_iou):
        return PartStatus(
            NEEDS_REVIEW,
            f"wilayah {part} tertutup pekerja lain - tidak dapat dinilai",
        )

    reason = (f"tidak ada APD terdeteksi pada {part} sementara wilayahnya "
              "terlihat jelas - kemungkinan pelanggaran, atau deteksi terlewat")

    if cfg.path_b_mode == "violation":
        return PartStatus(VIOLATION_INFERRED, reason)
    if cfg.path_b_mode == "review":
        return PartStatus(NEEDS_REVIEW, reason)
    return PartStatus(NEEDS_REVIEW, f"tidak ada APD terdeteksi pada {part}")


# --------------------------------------------------------------------------
# Penilaian
# --------------------------------------------------------------------------

def _assess_part(part: str, worker: Worker, all_persons: List[Detection],
                 img_w: int, img_h: int, cfg: InferenceConfig) -> PartStatus:
    det: Optional[Detection] = worker.head if part == "head" else worker.torso
    positive = HEAD_POSITIVE if part == "head" else TORSO_POSITIVE
    negative = HEAD_NEGATIVE if part == "head" else TORSO_NEGATIVE

    if det is None:
        return _assess_absent(part, worker, all_persons, img_w, img_h, cfg)

    if det.class_name == positive:
        return PartStatus(COMPLIANT, f"{positive} terdeteksi dan terasosiasi",
                          evidence=positive, confidence=round(det.confidence, 3))

    if det.class_name == negative:
        return PartStatus(VIOLATION, f"{negative} terdeteksi eksplisit (jalur A)",
                          evidence=negative, confidence=round(det.confidence, 3))

    return PartStatus(NEEDS_REVIEW, f"kelas tak terduga: {det.class_name}")


def _overall(head: PartStatus, torso: PartStatus) -> str:
    """
    Status gabungan. Urutan prioritas disengaja:
    pelanggaran mengalahkan perlu-tinjau, perlu-tinjau mengalahkan patuh.

    Seorang pekerja tidak pernah dinyatakan COMPLIANT bila ada bagian tubuh
    yang tidak dapat dinilai.
    """
    statuses = {head.status, torso.status}
    if VIOLATION in statuses:
        return VIOLATION
    if VIOLATION_INFERRED in statuses:
        return VIOLATION_INFERRED
    if NEEDS_REVIEW in statuses:
        return NEEDS_REVIEW
    return COMPLIANT


def assess(workers: List[Worker], orphans: List[Detection],
           img_w: int, img_h: int,
           cfg: InferenceConfig) -> Dict[str, object]:
    """Nilai kepatuhan seluruh pekerja pada satu gambar."""
    all_persons = [w.person for w in workers]
    results: List[WorkerCompliance] = []

    for w in workers:
        head = _assess_part("head", w, all_persons, img_w, img_h, cfg)
        torso = _assess_part("torso", w, all_persons, img_w, img_h, cfg)

        notes: List[str] = []
        for slot in w.ambiguous_parts:
            notes.append(f"APD {slot} dapat diklaim lebih dari satu pekerja - "
                         "dipilih berdasarkan skor tertinggi")

        results.append(WorkerCompliance(
            index=w.index,
            bbox=w.person.xyxy,
            person_confidence=round(w.person.confidence, 3),
            head=head, torso=torso,
            overall=_overall(head, torso),
            notes=notes,
        ))

    counts: Dict[str, int] = {}
    for r in results:
        counts[r.overall] = counts.get(r.overall, 0) + 1

    n = len(results)
    n_ok = counts.get(COMPLIANT, 0)
    n_violation = counts.get(VIOLATION, 0) + counts.get(VIOLATION_INFERRED, 0)
    n_review = counts.get(NEEDS_REVIEW, 0)

    summary = {
        "n_workers": n,
        "n_compliant": n_ok,
        "n_violation": n_violation,
        "n_needs_review": n_review,
        # Tingkat kepatuhan dihitung terhadap pekerja yang DAPAT dinilai.
        # Memasukkan NEEDS_REVIEW ke penyebut akan menghukum sistem karena
        # bersikap jujur soal ketidakpastian.
        "compliance_rate": round(n_ok / max(n_ok + n_violation, 1), 4),
        "assessable": n_ok + n_violation,
        "orphan_ppe": [
            {"class": o.class_name, "confidence": round(o.confidence, 3)}
            for o in orphans
        ],
    }

    if orphans:
        summary["orphan_warning"] = (
            f"{len(orphans)} box APD tidak menemukan pemilik. "
            "Kemungkinan ada pekerja yang tidak terdeteksi."
        )

    return {
        "workers": [r.to_dict() for r in results],
        "summary": summary,
        # keluaran gaya lama, sebagai pemenuhan requirement minimum panduan
        "class_counts": _class_counts(workers, orphans),
    }


def _class_counts(workers: List[Worker], orphans: List[Detection]) -> Dict[str, int]:
    """
    Penghitungan per kelas - format yang diminta contoh pada panduan.

    Disediakan sebagai pelengkap, bukan sebagai dasar penilaian kepatuhan.
    Lihat docstring modul association untuk alasannya.
    """
    counts: Dict[str, int] = {"person": len(workers)}
    for w in workers:
        for det in (w.head, w.torso):
            if det is not None:
                counts[det.class_name] = counts.get(det.class_name, 0) + 1
    for o in orphans:
        counts[o.class_name] = counts.get(o.class_name, 0) + 1
    return counts
