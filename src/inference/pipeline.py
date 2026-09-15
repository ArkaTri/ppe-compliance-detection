"""
Pipeline: dari gambar menjadi laporan kepatuhan per pekerja.

Satu titik masuk yang dipakai bersama oleh CLI dan aplikasi Streamlit,
sehingga logika penilaian tidak pernah bercabang antara keduanya.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .association import Detection, associate
from .compliance import assess
from .config import InferenceConfig

logger = logging.getLogger(__name__)

_MODEL_CACHE: Dict[str, object] = {}


def load_model(cfg: InferenceConfig):
    """Muat model sekali lalu simpan di cache - penting untuk Streamlit."""
    key = str(cfg.weights)
    if key not in _MODEL_CACHE:
        from ultralytics import YOLO
        logger.info("Memuat model: %s", key)
        _MODEL_CACHE[key] = YOLO(key)
    return _MODEL_CACHE[key]


def detect(image, cfg: InferenceConfig, model=None):
    """
    Jalankan detector pada satu gambar (path atau array RGB).

    conf sengaja diset ke threshold TERENDAH di antara seluruh kelas,
    lalu penyaringan per kelas dilakukan di tahap asosiasi. Dengan begitu
    kelas pelanggaran yang butuh threshold rendah tidak terbuang oleh
    satu nilai conf global.
    """
    model = model or load_model(cfg)

    if isinstance(image, (str, Path)):
        from PIL import Image
        arr = np.asarray(Image.open(image).convert("RGB"))
    elif hasattr(image, "convert"):
        arr = np.asarray(image.convert("RGB"))
    else:
        arr = np.asarray(image)

    from .cascade import detect_cascade
    return detect_cascade(arr, cfg, model)


def analyze(image, cfg: InferenceConfig, model=None,
            image_id: Optional[str] = None) -> Dict[str, object]:
    """Alur lengkap: deteksi -> asosiasi -> penilaian kepatuhan."""
    if isinstance(image, (str, Path)):
        from PIL import Image
        with Image.open(image) as im:
            img_w, img_h = im.size
        image_id = image_id or Path(image).stem
    else:
        arr = np.asarray(image)
        img_h, img_w = arr.shape[:2]

    detections, det_stats = detect(image, cfg, model=model)
    workers, orphans = associate(detections, cfg)
    report = assess(workers, orphans, img_w, img_h, cfg)

    report["image_id"] = image_id
    report["image_size"] = [img_w, img_h]
    report["_workers"] = workers      # untuk anotasi
    report["_orphans"] = orphans
    report["_detections"] = detections
    report["detection_stats"] = det_stats

    # Penilaian ketidaksesuaian domain - tidak mengubah vonis per pekerja,
    # hanya menambahkan konteks bila gambar tampak di luar wilayah latih.
    from .domain_check import assess_domain
    report["domain"] = assess_domain(report)

    return report
