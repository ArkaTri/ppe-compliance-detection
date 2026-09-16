"""
Fungsi pencocokan dan metrik - dipakai bersama oleh kalibrasi dan validasi.

Dipisahkan dari `src/training/evaluate.py` agar modul inferensi tidak menarik
dependensi paket training, dan agar logika pencocokan hanya punya satu sumber
kebenaran.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Hitung IoU antara semua pasangan box A dan B sekaligus (vectorized),
    jauh lebih cepat daripada loop per pasangan saat validasi ratusan gambar.
    """
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=float)
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return np.where(union > 0, inter / union, 0.0)


def match_image(pred_boxes, pred_cls, pred_conf, gt_boxes, gt_cls,
                iou_thr: float = 0.5) -> List[dict]:
    """Cocokkan prediksi ke ground truth, greedy menurut keyakinan."""
    records: List[dict] = []
    classes = set(np.asarray(pred_cls).tolist()) | set(np.asarray(gt_cls).tolist())

    for c in sorted(classes):
        pm, gm = pred_cls == c, gt_cls == c
        pb, pc = pred_boxes[pm], pred_conf[pm]
        gb = gt_boxes[gm]

        order = np.argsort(-pc)
        pb, pc = pb[order], pc[order]

        ious = iou_matrix(pb, gb)
        taken = np.zeros(len(gb), dtype=bool)
        tp_conf, fp_conf = [], []

        for i in range(len(pb)):
            cand = np.where(~taken)[0] if len(gb) else np.array([], int)
            if len(cand) == 0:
                fp_conf.append(float(pc[i]))
                continue
            j = cand[np.argmax(ious[i, cand])]
            if ious[i, j] >= iou_thr:
                taken[j] = True
                tp_conf.append(float(pc[i]))
            else:
                fp_conf.append(float(pc[i]))

        records.append({"class_id": int(c), "n_gt": int(len(gb)),
                        "tp_conf": tp_conf, "fp_conf": fp_conf})
    return records


def metrics_at(per_image: List[dict], class_id: int, thr: float) -> Dict[str, float]:
    """Precision / recall / F1 / F2 satu kelas pada ambang tertentu."""
    tp = fp = n_gt = 0
    for img in per_image:
        for rec in img["records"]:
            if rec["class_id"] != class_id:
                continue
            n_gt += rec["n_gt"]
            tp += sum(1 for c in rec["tp_conf"] if c >= thr)
            fp += sum(1 for c in rec["fp_conf"] if c >= thr)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / n_gt if n_gt else 0.0

    # F-beta: beta=1 seimbangkan precision & recall, beta=2 bobot recall 2x lipat.
    def fbeta(beta: float) -> float:
        b2 = beta * beta
        d = b2 * precision + recall
        return (1 + b2) * precision * recall / d if d else 0.0

    return {"threshold": round(thr, 3), "tp": tp, "fp": fp,
            "fn": n_gt - tp, "n_gt": n_gt,
            "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(fbeta(1.0), 4), "f2": round(fbeta(2.0), 4)}


def sweep_thresholds(per_image: List[dict], class_id: int,
                     start: float = 0.05, stop: float = 0.90,
                     step: float = 0.05) -> List[Dict[str, float]]:
    """Hitung precision/recall pada rentang threshold untuk mencari titik optimal."""
    return [metrics_at(per_image, class_id, float(t))
            for t in np.arange(start, stop + 1e-9, step)]
