"""
Association engine: memetakan setiap box APD ke pekerja pemiliknya.

Ini inti pembeda proyek. Contoh pada panduan capstone memakai penghitungan
kelas: "4 pekerja, 4 helmet, 2 vest, 2 no-vest, berarti 2 orang tidak
lengkap". Kesimpulan itu hanya benar secara kebetulan - penghitungan kelas
tidak dapat menentukan SIAPA yang melanggar.

Contoh sanggahan dari dataset ini: gambar ppe_0683 (terowongan) memuat lebih
banyak box helmet daripada box person. Logika count-based langsung runtuh -
dan runtuh secara diam-diam, karena keluarannya tetap terlihat masuk akal.

Metode:
  1. containment = area(irisan) / area(box APD)  -- BUKAN IoU.
     Box helm luasnya bisa hanya 5% dari box tubuh, sehingga IoU maksimalnya
     juga sekitar 0.05 meski helm berada sepenuhnya di dalam bbox person.
     Memakai IoU standar membuat seluruh asosiasi gagal.
  2. bonus zona anatomis bila APD berada di bagian tubuh yang diharapkan.
  3. penugasan greedy berdasarkan skor tertinggi.

Greedy dipilih, bukan Hungarian. EDA menunjukkan ambiguity rate hanya 0.6%
pada ground truth - Hungarian hanya menambah kompleksitas tanpa memperbaiki
apa pun. Namun penugasan TETAP berbasis skor, bukan pencocokan pertama yang
ditemukan, karena ambiguitas pada prediksi lebih tinggi daripada pada
ground truth (anotator kerap melewatkan person di adegan padat).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import (
    HEAD_CLASSES,
    InferenceConfig,
    PERSON,
    PPE_CLASSES,
    TORSO_CLASSES,
)


@dataclass
class Detection:
    """Satu kotak deteksi."""
    class_name: str
    confidence: float
    xyxy: Tuple[float, float, float, float]

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    @property
    def center_y(self) -> float:
        return (self.xyxy[1] + self.xyxy[3]) / 2


@dataclass
class Worker:
    """Seorang pekerja beserta APD yang terasosiasi dengannya."""
    index: int
    person: Detection
    head: Optional[Detection] = None
    torso: Optional[Detection] = None
    head_score: float = 0.0
    torso_score: float = 0.0
    ambiguous_parts: List[str] = field(default_factory=list)


def containment(ppe: Detection, person: Detection) -> float:
    """Fraksi luas box APD yang berada di dalam box person."""
    px1, py1, px2, py2 = ppe.xyxy
    qx1, qy1, qx2, qy2 = person.xyxy

    iw = min(px2, qx2) - max(px1, qx1)
    ih = min(py2, qy2) - max(py1, qy1)
    if iw <= 0 or ih <= 0:
        return 0.0
    if ppe.area <= 0:
        return 0.0
    return float((iw * ih) / ppe.area)


def relative_y(ppe: Detection, person: Detection) -> float:
    """Posisi pusat APD terhadap tinggi person: 0 = ubun-ubun, 1 = kaki."""
    py1, py2 = person.xyxy[1], person.xyxy[3]
    height = py2 - py1
    if height <= 0:
        return float("nan")
    return float((ppe.center_y - py1) / height)


def association_score(ppe: Detection, person: Detection,
                      cfg: InferenceConfig) -> float:
    """
    Skor asosiasi = containment + bonus zona.

    Bonus diberikan, bukan syarat. Constraint geometris yang keras akan
    membuang asosiasi yang benar pada postur jongkok atau membungkuk -
    postur yang justru lazim di lokasi konstruksi.
    """
    c = containment(ppe, person)
    if c < cfg.containment_threshold:
        return 0.0

    ry = relative_y(ppe, person)
    if np.isnan(ry):
        return c

    zone = cfg.head_zone if ppe.class_name in HEAD_CLASSES else cfg.torso_zone
    if zone[0] <= ry <= zone[1]:
        return c + cfg.zone_bonus
    return c


def filter_by_threshold(detections: List[Detection],
                        cfg: InferenceConfig) -> List[Detection]:
    """
    Terapkan threshold per kelas.

    Kelas pelanggaran memakai threshold lebih rendah karena melewatkan
    pelanggaran (false negative) jauh lebih mahal daripada alarm palsu
    dalam konteks keselamatan kerja.
    """
    return [d for d in detections
            if d.confidence >= cfg.threshold(d.class_name)]


def associate(detections: List[Detection],
              cfg: InferenceConfig) -> Tuple[List[Worker], List[Detection]]:
    """
    Petakan APD ke pekerja.

    Return (daftar Worker, daftar APD yatim yang tidak menemukan pemilik).
    APD yatim tidak dibuang diam-diam - ia dilaporkan, karena bisa berarti
    ada pekerja yang tidak terdeteksi.
    """
    dets = filter_by_threshold(detections, cfg)

    persons = [d for d in dets if d.class_name == PERSON]
    ppes = [d for d in dets if d.class_name in PPE_CLASSES]

    # urut dari kiri ke kanan supaya penomoran pekerja stabil dan mudah
    # dirujuk saat menonton gambar
    persons.sort(key=lambda d: d.xyxy[0])
    workers = [Worker(index=i + 1, person=p) for i, p in enumerate(persons)]

    # kumpulkan seluruh kandidat (skor, worker, ppe)
    candidates: List[Tuple[float, int, int]] = []
    ambiguity: Dict[int, int] = {}

    for j, ppe in enumerate(ppes):
        n_candidates = 0
        for i, w in enumerate(workers):
            s = association_score(ppe, w.person, cfg)
            if s > 0:
                candidates.append((s, i, j))
                n_candidates += 1
        ambiguity[j] = n_candidates

    # greedy: skor tertinggi lebih dulu
    candidates.sort(key=lambda t: -t[0])

    used_ppe: set = set()
    for score, wi, pj in candidates:
        if pj in used_ppe:
            continue
        ppe = ppes[pj]
        worker = workers[wi]
        slot = "head" if ppe.class_name in HEAD_CLASSES else "torso"

        if getattr(worker, slot) is not None:
            continue  # slot sudah terisi oleh kandidat berskor lebih tinggi

        setattr(worker, slot, ppe)
        setattr(worker, f"{slot}_score", round(score, 4))
        if ambiguity.get(pj, 0) > 1:
            worker.ambiguous_parts.append(slot)
        used_ppe.add(pj)

    orphans = [ppes[j] for j in range(len(ppes)) if j not in used_ppe]
    return workers, orphans
