"""
Evaluasi: metrik per kelas dengan ketidakpastian yang eksplisit.

Mengapa tidak cukup memakai `model.val()` bawaan Ultralytics:

  1. Ia melaporkan satu angka per kelas tanpa interval kepercayaan. Untuk
     `no-helmet` yang hanya punya 11 instance di validation, satu deteksi
     meleset menggeser recall sekitar 9 poin. Angka tunggal menyembunyikan
     ketidakpastian sebesar itu.
  2. Ia tidak menyediakan sapuan threshold per kelas. Padahal di sistem
     keselamatan, kelas pelanggaran perlu threshold lebih rendah daripada
     kelas aman - false negative jauh lebih mahal daripada false positive.

Modul ini melakukan pencocokan prediksi-ground truth sendiri sehingga kedua
hal di atas bisa dihitung.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# --------------------------------------------------------------------------
# Pencocokan
# --------------------------------------------------------------------------

def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU antara dua himpunan box format xyxy. Return matriks (len(a), len(b))."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=float)

    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]

    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = iw * ih

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return np.where(union > 0, inter / union, 0.0)


def match_image(pred_boxes: np.ndarray, pred_cls: np.ndarray, pred_conf: np.ndarray,
                gt_boxes: np.ndarray, gt_cls: np.ndarray,
                iou_thr: float = 0.5) -> List[dict]:
    """
    Cocokkan prediksi ke ground truth pada satu gambar, greedy by confidence.

    Return daftar record per kelas: {class_id, tp, fp, fn, conf_of_tp, conf_of_fp}.
    Skor confidence disimpan agar sapuan threshold bisa dilakukan belakangan
    tanpa menjalankan ulang model.
    """
    records: List[dict] = []
    classes = set(pred_cls.tolist()) | set(gt_cls.tolist())

    for c in sorted(classes):
        p_mask = pred_cls == c
        g_mask = gt_cls == c
        pb, pc = pred_boxes[p_mask], pred_conf[p_mask]
        gb = gt_boxes[g_mask]

        order = np.argsort(-pc)
        pb, pc = pb[order], pc[order]

        ious = iou_matrix(pb, gb)
        taken = np.zeros(len(gb), dtype=bool)
        tp_conf, fp_conf = [], []

        for i in range(len(pb)):
            if len(gb) == 0:
                fp_conf.append(float(pc[i]))
                continue
            cand = np.where(~taken)[0]
            if len(cand) == 0:
                fp_conf.append(float(pc[i]))
                continue
            j = cand[np.argmax(ious[i, cand])]
            if ious[i, j] >= iou_thr:
                taken[j] = True
                tp_conf.append(float(pc[i]))
            else:
                fp_conf.append(float(pc[i]))

        records.append({
            "class_id": int(c),
            "n_gt": int(len(gb)),
            "tp_conf": tp_conf,
            "fp_conf": fp_conf,
        })
    return records


# --------------------------------------------------------------------------
# Inferensi atas satu split
# --------------------------------------------------------------------------

def collect_matches(model, dataset_root: Path, split: str,
                    conf_floor: float = 0.001,
                    iou_thr: float = 0.5) -> List[dict]:
    """
    Jalankan model pada satu split dan kumpulkan hasil pencocokan per gambar.

    `conf_floor` sengaja sangat rendah supaya seluruh kurva threshold bisa
    disapu belakangan dari satu kali inferensi.
    """
    img_dir = dataset_root / split / "images"
    lbl_dir = dataset_root / split / "labels"
    per_image: List[dict] = []

    images = sorted(p for p in img_dir.iterdir()
                    if p.suffix.lower() in IMAGE_EXTS)
    logger.info("Inferensi pada split '%s': %d gambar", split, len(images))

    for img_path in images:
        result = model(str(img_path), conf=conf_floor, verbose=False)[0]
        h, w = result.orig_shape

        if result.boxes is not None and len(result.boxes) > 0:
            pb = result.boxes.xyxy.cpu().numpy()
            pc = result.boxes.cls.cpu().numpy().astype(int)
            pf = result.boxes.conf.cpu().numpy()
        else:
            pb = np.zeros((0, 4)); pc = np.zeros(0, int); pf = np.zeros(0)

        gb, gc = _load_gt(lbl_dir / f"{img_path.stem}.txt", w, h)
        per_image.append({
            "image_id": img_path.stem,
            "records": match_image(pb, pc, pf, gb, gc, iou_thr=iou_thr),
        })

    return per_image


def _load_gt(label_path: Path, w: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
    """Baca label YOLO ternormalisasi menjadi box xyxy absolut."""
    if not label_path.exists():
        return np.zeros((0, 4)), np.zeros(0, int)

    boxes, classes = [], []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        c = int(float(parts[0]))
        xc, yc, bw, bh = (float(v) for v in parts[1:5])
        boxes.append([(xc - bw / 2) * w, (yc - bh / 2) * h,
                      (xc + bw / 2) * w, (yc + bh / 2) * h])
        classes.append(c)

    if not boxes:
        return np.zeros((0, 4)), np.zeros(0, int)
    return np.asarray(boxes, dtype=float), np.asarray(classes, dtype=int)


# --------------------------------------------------------------------------
# Metrik pada threshold tertentu
# --------------------------------------------------------------------------

def metrics_at(per_image: List[dict], class_id: int, thr: float) -> Dict[str, float]:
    """Precision / recall / F1 / F2 satu kelas pada threshold tertentu."""
    tp = fp = n_gt = 0
    for img in per_image:
        for rec in img["records"]:
            if rec["class_id"] != class_id:
                continue
            n_gt += rec["n_gt"]
            tp += sum(1 for c in rec["tp_conf"] if c >= thr)
            fp += sum(1 for c in rec["fp_conf"] if c >= thr)

    fn = n_gt - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / n_gt if n_gt else 0.0

    # F-beta: beta=1 seimbang, beta=2 membobot recall (dipakai untuk kelas pelanggaran).
    def fbeta(beta: float) -> float:
        b2 = beta * beta
        denom = b2 * precision + recall
        return (1 + b2) * precision * recall / denom if denom else 0.0

    return {
        "threshold": round(thr, 3),
        "tp": tp, "fp": fp, "fn": fn, "n_gt": n_gt,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(fbeta(1.0), 4),
        "f2": round(fbeta(2.0), 4),  # membobot recall 2x lipat
    }


def sweep_thresholds(per_image: List[dict], class_id: int,
                     start: float = 0.05, stop: float = 0.90,
                     step: float = 0.05) -> List[Dict[str, float]]:
    """Hitung precision/recall pada rentang threshold, dasar penentuan threshold optimal."""
    thrs = np.arange(start, stop + 1e-9, step)
    return [metrics_at(per_image, class_id, float(t)) for t in thrs]


def recommend_threshold(per_image: List[dict], class_id: int,
                        prioritize_recall: bool) -> Dict[str, float]:
    """
    Pilih threshold optimal untuk satu kelas.

    Kelas pelanggaran dioptimalkan pada F2 (recall dibobot 2x), kelas lain
    pada F1. Inilah dasar kuantitatif untuk threshold asimetris - bukan
    angka yang ditebak.
    """
    sweep = sweep_thresholds(per_image, class_id)
    key = "f2" if prioritize_recall else "f1"
    best = max(sweep, key=lambda r: r[key])
    return {"optimized_for": key, **best}


# --------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------

def bootstrap_metric(per_image: List[dict], class_id: int, thr: float,
                     metric: str = "recall", n_boot: int = 2000,
                     seed: int = 42) -> Dict[str, float]:
    """
    Interval kepercayaan 95% via bootstrap atas GAMBAR (bukan atas box).

    Resampling dilakukan pada level gambar karena box di dalam satu gambar
    tidak independen - pekerja pada satu foto berbagi kondisi pencahayaan,
    jarak kamera, dan konteks yang sama.
    """
    rng = np.random.default_rng(seed)
    n = len(per_image)
    if n == 0:
        return {}

    idx_all = np.arange(n)
    vals: List[float] = []

    for _ in range(n_boot):
        sample = [per_image[i] for i in rng.choice(idx_all, size=n, replace=True)]
        m = metrics_at(sample, class_id, thr)
        if m["n_gt"] > 0:
            vals.append(m[metric])

    if not vals:
        return {}

    arr = np.asarray(vals)
    point = metrics_at(per_image, class_id, thr)[metric]
    return {
        "metric": metric,
        "point_estimate": round(float(point), 4),
        "ci95_low": round(float(np.percentile(arr, 2.5)), 4),
        "ci95_high": round(float(np.percentile(arr, 97.5)), 4),
        "ci_width": round(float(np.percentile(arr, 97.5) -
                                np.percentile(arr, 2.5)), 4),
        "n_bootstrap": len(vals),
    }
