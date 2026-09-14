"""
Validasi logika kepatuhan terhadap ground truth.

Modul ini menjawab pertanyaan yang tidak dijawab oleh metrik deteksi biasa:
seberapa sering keputusan KEPATUHAN sistem benar?

mAP dan recall mengukur kualitas kotak. Yang sampai ke pengguna bukan kotak,
melainkan vonis per pekerja: patuh, melanggar, atau perlu ditinjau. Vonis itu
yang harus diukur.

Yang paling penting diukur di sini adalah presisi JALUR B - pelanggaran yang
disimpulkan dari ketiadaan APD. Jalur ini ada karena detector `no-helmet`
lemah (recall 0.542). Tetapi bila jalur B sendiri sering salah, ia hanya
mengganti satu kegagalan dengan kegagalan lain yang lebih berbahaya:
menuduh pekerja yang sebenarnya patuh.

Metode:
  1. Bangun laporan kepatuhan dari PREDIKSI model.
  2. Bangun laporan kepatuhan dari GROUND TRUTH, memakai logika asosiasi
     yang sama persis.
  3. Cocokkan pekerja prediksi ke pekerja ground truth via IoU.
  4. Bandingkan vonis per bagian tubuh.

Pekerja ground truth yang tidak punya label APD sama sekali pada suatu
bagian tubuh dikeluarkan dari penilaian bagian itu - anotasi tidak memberi
tahu apakah orang tersebut patuh atau tidak, jadi menilainya akan menghukum
sistem atas kekosongan anotasi.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .association import Detection, associate
from .compliance import assess
from .config import (
    COMPLIANT,
    HEAD_CLASSES,
    HEAD_NEGATIVE,
    HEAD_POSITIVE,
    InferenceConfig,
    NEEDS_REVIEW,
    TORSO_NEGATIVE,
    TORSO_POSITIVE,
    VIOLATION,
    VIOLATION_INFERRED,
)
from .pipeline import analyze

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
UNKNOWN = "UNKNOWN"


# --------------------------------------------------------------------------
# Ground truth
# --------------------------------------------------------------------------

def load_gt_detections(label_path: Path, img_w: int, img_h: int,
                       class_names: List[str]) -> List[Detection]:
    """Baca label YOLO menjadi Detection dengan confidence 1.0."""
    if not label_path.exists():
        return []

    out: List[Detection] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cid = int(float(parts[0]))
        if not 0 <= cid < len(class_names):
            continue
        xc, yc, bw, bh = (float(v) for v in parts[1:5])
        out.append(Detection(
            class_name=class_names[cid],
            confidence=1.0,
            xyxy=((xc - bw / 2) * img_w, (yc - bh / 2) * img_h,
                  (xc + bw / 2) * img_w, (yc + bh / 2) * img_h),
        ))
    return out


def gt_part_status(worker, part: str) -> str:
    """
    Vonis ground truth untuk satu bagian tubuh.

    UNKNOWN bila anotator tidak melabeli APD apa pun di sana - bukan berarti
    pekerja melanggar, melainkan anotasinya memang kosong.
    """
    det = worker.head if part == "head" else worker.torso
    if det is None:
        return UNKNOWN
    positive = HEAD_POSITIVE if part == "head" else TORSO_POSITIVE
    return COMPLIANT if det.class_name == positive else VIOLATION


def iou(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


# --------------------------------------------------------------------------
# Validasi
# --------------------------------------------------------------------------

def validate_image(image_path: Path, label_path: Path,
                   class_names: List[str], cfg: InferenceConfig,
                   model=None, match_iou: float = 0.5) -> List[dict]:
    """Bandingkan vonis prediksi vs ground truth untuk satu gambar."""
    from PIL import Image
    with Image.open(image_path) as im:
        img_w, img_h = im.size

    pred = analyze(str(image_path), cfg, model=model)
    pred_workers = pred["_workers"]

    gt_dets = load_gt_detections(label_path, img_w, img_h, class_names)
    gt_workers, gt_orphans = associate(gt_dets, cfg)

    rows: List[dict] = []
    used_gt: set = set()

    for pw, pinfo in zip(pred_workers, pred["workers"]):
        best_j, best_iou = -1, 0.0
        for j, gw in enumerate(gt_workers):
            if j in used_gt:
                continue
            v = iou(pw.person.xyxy, gw.person.xyxy)
            if v > best_iou:
                best_iou, best_j = v, j

        if best_iou < match_iou:
            rows.append({
                "image_id": image_path.stem,
                "matched": False,
                "note": "pekerja terdeteksi tidak ada padanannya di ground truth",
            })
            continue

        used_gt.add(best_j)
        gw = gt_workers[best_j]
        for part in ("head", "torso"):
            rows.append({
                "image_id": image_path.stem,
                "matched": True,
                "part": part,
                "iou": round(best_iou, 3),
                "pred": pinfo[part]["status"],
                "gt": gt_part_status(gw, part),
            })

    missed = len(gt_workers) - len(used_gt)
    if missed > 0:
        rows.append({
            "image_id": image_path.stem,
            "matched": False,
            "note": f"{missed} pekerja di ground truth tidak terdeteksi model",
        })
    return rows


def validate_split(dataset_root: Path, split: str, class_names: List[str],
                   cfg: InferenceConfig, limit: Optional[int] = None) -> Dict[str, object]:
    """Jalankan validasi pada seluruh gambar dalam satu split."""
    from .pipeline import load_model
    model = load_model(cfg)

    img_dir = Path(dataset_root) / split / "images"
    lbl_dir = Path(dataset_root) / split / "labels"
    images = sorted(p for p in img_dir.iterdir()
                    if p.suffix.lower() in IMAGE_EXTS)
    if limit:
        images = images[:limit]

    all_rows: List[dict] = []
    for i, img in enumerate(images, 1):
        if i % 20 == 0:
            logger.info("  %d/%d gambar", i, len(images))
        all_rows.extend(validate_image(img, lbl_dir / f"{img.stem}.txt",
                                       class_names, cfg, model=model))

    return summarize(all_rows)


def summarize(rows: List[dict]) -> Dict[str, object]:
    """Ringkas menjadi metrik keputusan kepatuhan."""
    graded = [r for r in rows
              if r.get("matched") and r.get("gt") not in (None, UNKNOWN)]

    def count(pred_status: str, gt_status: str) -> int:
        return sum(1 for r in graded
                   if r["pred"] == pred_status and r["gt"] == gt_status)

    def precision(pred_status: str, correct_gt: str) -> Optional[float]:
        total = sum(1 for r in graded if r["pred"] == pred_status)
        if total == 0:
            return None
        return round(count(pred_status, correct_gt) / total, 4)

    n_gt_violation = sum(1 for r in graded if r["gt"] == VIOLATION)
    caught_a = count(VIOLATION, VIOLATION)
    caught_b = count(VIOLATION_INFERRED, VIOLATION)

    # semua baris, termasuk yang gt-nya UNKNOWN, untuk menghitung proporsi
    parts = [r for r in rows if r.get("matched")]
    n_parts = max(len(parts), 1)

    out = {
        "n_part_decisions": len(parts),
        "n_graded": len(graded),
        "n_excluded_unknown_gt": len(parts) - len(graded),

        "path_a": {
            "n_predicted": sum(1 for r in graded if r["pred"] == VIOLATION),
            "precision": precision(VIOLATION, VIOLATION),
            "caught": caught_a,
        },
        "path_b": {
            "n_predicted": sum(1 for r in graded if r["pred"] == VIOLATION_INFERRED),
            "precision": precision(VIOLATION_INFERRED, VIOLATION),
            "caught": caught_b,
        },
        "compliant_precision": precision(COMPLIANT, COMPLIANT),

        "violation_recall": {
            "n_gt_violations": n_gt_violation,
            "path_a_only": round(caught_a / max(n_gt_violation, 1), 4),
            "combined": round((caught_a + caught_b) / max(n_gt_violation, 1), 4),
            "path_b_contribution": caught_b,
        },

        "needs_review_rate": round(
            sum(1 for r in parts if r["pred"] == NEEDS_REVIEW) / n_parts, 4),

        "unmatched_notes": [r["note"] for r in rows if not r.get("matched")][:20],
    }

    # matriks kebingungan vonis
    matrix: Dict[str, Dict[str, int]] = {}
    for r in graded:
        matrix.setdefault(r["pred"], {})
        matrix[r["pred"]][r["gt"]] = matrix[r["pred"]].get(r["gt"], 0) + 1
    out["confusion"] = matrix

    return out
