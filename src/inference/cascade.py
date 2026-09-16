"""
Deteksi kaskade dua tahap - memakai detector terkuat untuk menolong yang terlemah.

Masalah yang diserang:
    Median luas box helm hanya 0.0093 dari luas gambar (sekitar 62 px pada
    frame 960), sementara box person mencapai 0.1049. Detector bekerja pada
    peta fitur yang sudah diperkecil beberapa kali, sehingga objek sekecil
    helm hanya menempati beberapa sel. Kelas `no-helmet` lebih buruk lagi:
    median 0.0062, dan hanya 129 instance di seluruh dataset.

Gagasan:
    Tahap 1 mendeteksi `person` pada gambar penuh. Ini kekuatan model -
    2817 instance pelatihan, recall 0.846 pada test set.

    Tahap 2 memotong setiap bbox person, lalu menjalankan detector yang SAMA
    pada potongan tersebut. Potongan diperbesar ke resolusi inferensi penuh,
    sehingga helm yang tadinya 62 px menjadi ratusan piksel. Peningkatan
    resolusi efektif sekitar 5-8x pada kelas yang paling membutuhkannya.

    Tidak ada model baru dan tidak ada pelatihan ulang - hanya cara memakai
    model yang sudah ada secara berbeda.

Penggabungan hasil:
    Deteksi dari kedua tahap digabung dengan NMS per GRUP EKSKLUSIF, bukan
    per kelas. `helmet` dan `no-helmet` saling meniadakan pada satu kepala -
    keduanya tidak boleh bertahan bersama. NMS per kelas biasa akan
    meloloskan keduanya dan menghasilkan vonis yang bertentangan.

Biaya:
    N+1 pemanggilan model untuk N pekerja. Pada CPU sekitar 0.5 detik per
    panggilan, sehingga 5 pekerja memakan ~3 detik. Dibatasi oleh
    `max_crops` agar gambar kerumunan tidak meledak waktu prosesnya.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .association import Detection
from .config import (
    HEAD_CLASSES,
    InferenceConfig,
    PERSON,
    PPE_CLASSES,
    TORSO_CLASSES,
)

logger = logging.getLogger(__name__)

# Grup yang saling meniadakan: satu kepala tidak bisa sekaligus
# memakai dan tidak memakai helm.
EXCLUSIVE_GROUPS: Tuple[Tuple[str, ...], ...] = (HEAD_CLASSES, TORSO_CLASSES)

def _to_bgr(arr: np.ndarray) -> np.ndarray:
    """
    Ultralytics memperlakukan array numpy sebagai BGR (konvensi OpenCV).
    Pipeline ini bekerja dalam RGB karena anotasi memakai Pillow, sehingga
    kanal harus dibalik tepat sebelum pemanggilan model.

    Tanpa pembalikan ini, model menerima gambar dengan kanal merah dan biru
    tertukar: presisi vonis pelanggaran turun dari 0.89 ke 0.64 tanpa gejala
    error apa pun.
    """
    return np.ascontiguousarray(arr[..., ::-1])

def _group_of(class_name: str) -> Optional[int]:
    """Cari indeks grup eksklusif (head/torso) tempat sebuah kelas berada."""
    for i, group in enumerate(EXCLUSIVE_GROUPS):
        if class_name in group:
            return i
    return None


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection-over-Union standar — dipakai untuk NMS antar grup eksklusif."""
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def merge_exclusive(detections: List[Detection],
                    iou_thr: float = 0.55) -> List[Detection]:
    """
    NMS per grup eksklusif.

    Di dalam satu grup (kepala atau torso), box yang tumpang tindih melebihi
    ambang dianggap merujuk bagian tubuh yang sama - hanya yang berkeyakinan
    tertinggi dipertahankan, TERMASUK bila kelasnya berbeda. Inilah bedanya
    dengan NMS per kelas: `helmet` 0.50 dan `no-helmet` 0.72 pada kepala yang
    sama akan diselesaikan menjadi `no-helmet`, bukan dibiarkan keduanya.
    """
    kept: List[Detection] = []
    for det in sorted(detections, key=lambda d: -d.confidence):
        g = _group_of(det.class_name)
        conflict = False
        for other in kept:
            if _group_of(other.class_name) != g:
                continue
            if iou(det.xyxy, other.xyxy) >= iou_thr:
                conflict = True
                break
        if not conflict:
            kept.append(det)
    return kept


def _expand_box(box: Sequence[float], pad: float,
                img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    """
    Perlebar bbox person sebelum dipotong.

    Padding penting karena bbox person kerap memotong tepat di batas kepala,
    sehingga helm terpotong sebagian. Konteks di sekeliling juga membantu
    detector mengenali bentuk.
    """
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    dx, dy = w * pad, h * pad
    return (
        max(0, int(x1 - dx)),
        max(0, int(y1 - dy)),
        min(img_w, int(x2 + dx)),
        min(img_h, int(y2 + dy)),
    )


def detect_cascade(image_rgb: np.ndarray, cfg: InferenceConfig,
                   model) -> Tuple[List[Detection], Dict[str, object]]:
    """
    Jalankan deteksi dua tahap.

    Return (daftar deteksi gabungan, statistik untuk pelaporan).
    `image_rgb` adalah array HxWx3 dalam urutan RGB.
    """
    img_h, img_w = image_rgb.shape[:2]
    names = None

    # ---------------- Tahap 1: gambar penuh ----------------
    result = model(_to_bgr(image_rgb), imgsz=cfg.imgsz, conf=cfg.min_threshold,
                   device=cfg.device, verbose=False,
                   augment=cfg.use_tta)[0]
    names = result.names

    stage1: List[Detection] = []
    if result.boxes is not None and len(result.boxes) > 0:
        for b, c, f in zip(result.boxes.xyxy.cpu().numpy(),
                           result.boxes.cls.cpu().numpy().astype(int),
                           result.boxes.conf.cpu().numpy()):
            stage1.append(Detection(names[int(c)], float(f),
                                    (float(b[0]), float(b[1]),
                                     float(b[2]), float(b[3]))))

    persons = [d for d in stage1
               if d.class_name == PERSON and d.confidence >= cfg.threshold(PERSON)]
    ppe_stage1 = [d for d in stage1 if d.class_name in PPE_CLASSES]

    if not cfg.use_cascade:
        return stage1, {"cascade": False, "n_crops": 0}

    # ---------------- Tahap 2: potongan per pekerja ----------------
    # Pekerja terbesar didahulukan: mereka paling mungkin memuat APD yang
    # terlewat, dan bila `max_crops` membatasi, yang terpenting sudah diproses.
    persons_sorted = sorted(persons, key=lambda d: -d.area)[:cfg.max_crops]

    ppe_stage2: List[Detection] = []
    n_crops = 0
    n_recovered = 0

    for person in persons_sorted:
        cx1, cy1, cx2, cy2 = _expand_box(person.xyxy, cfg.crop_pad, img_w, img_h)
        cw, ch = cx2 - cx1, cy2 - cy1

        # Potongan yang sudah besar tidak akan mendapat keuntungan resolusi,
        # dan potongan terlalu kecil justru menghasilkan artefak pembesaran.
        if cw < cfg.min_crop_size or ch < cfg.min_crop_size:
            continue
        if cw >= img_w * cfg.skip_crop_ratio and ch >= img_h * cfg.skip_crop_ratio:
            continue

        crop = image_rgb[cy1:cy2, cx1:cx2]
        n_crops += 1

        r = model(_to_bgr(crop), imgsz=cfg.crop_imgsz, conf=cfg.min_threshold,
                  device=cfg.device, verbose=False, augment=cfg.use_tta)[0]
        if r.boxes is None or len(r.boxes) == 0:
            continue

        for b, c, f in zip(r.boxes.xyxy.cpu().numpy(),
                           r.boxes.cls.cpu().numpy().astype(int),
                           r.boxes.conf.cpu().numpy()):
            cls_name = names[int(c)]
            if cls_name not in PPE_CLASSES:
                continue  # person pada potongan tidak berguna
            # petakan kembali ke koordinat gambar asal - hanya pergeseran,
            # karena Ultralytics mengembalikan koordinat dalam dimensi potongan
            mapped = (float(b[0]) + cx1, float(b[1]) + cy1,
                      float(b[2]) + cx1, float(b[3]) + cy1)
            ppe_stage2.append(Detection(cls_name, float(f), mapped))

    # ---------------- Penggabungan ----------------
    before = len(ppe_stage1)
    merged_ppe = merge_exclusive(ppe_stage1 + ppe_stage2, cfg.merge_iou)

    # hitung berapa APD yang hanya ditemukan tahap 2
    for det in merged_ppe:
        if not any(iou(det.xyxy, s.xyxy) >= cfg.merge_iou
                   and _group_of(s.class_name) == _group_of(det.class_name)
                   for s in ppe_stage1):
            n_recovered += 1

    stats = {
        "cascade": True,
        "n_crops": n_crops,
        "n_ppe_stage1": before,
        "n_ppe_stage2_raw": len(ppe_stage2),
        "n_ppe_merged": len(merged_ppe),
        "n_recovered_by_cascade": n_recovered,
        "tta": cfg.use_tta,
    }
    logger.debug("Kaskade: %d potongan, %d APD baru ditemukan",
                 n_crops, n_recovered)

    return persons + merged_ppe, stats
