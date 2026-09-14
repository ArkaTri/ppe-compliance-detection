"""
Anotasi visual: menggambar vonis kepatuhan PER PEKERJA.

Berbeda dari anotasi object detection biasa yang mewarnai kotak menurut
kelasnya, modul ini mewarnai bbox person menurut STATUS KEPATUHANNYA.
Yang perlu dilihat safety officer bukan "ada 4 helm di gambar ini",
melainkan "pekerja #2 perlu ditinjau".

Memakai Pillow, bukan OpenCV. Alasannya praktis: opencv-python menarik
dependensi sistem yang kerap gagal di Streamlit Community Cloud, dan
seluruh kebutuhan di sini hanya menggambar kotak dan teks.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import (
    COMPLIANT,
    NEEDS_REVIEW,
    VIOLATION,
    VIOLATION_INFERRED,
)

# Warna per status. Merah hanya untuk pelanggaran yang benar-benar
# diputuskan; kuning untuk yang perlu tinjau manusia.
STATUS_COLOR: Dict[str, Tuple[int, int, int]] = {
    COMPLIANT: (34, 197, 94),            # hijau
    VIOLATION: (220, 38, 38),            # merah
    VIOLATION_INFERRED: (249, 115, 22),  # jingga
    NEEDS_REVIEW: (234, 179, 8),         # kuning
}

STATUS_LABEL: Dict[str, str] = {
    COMPLIANT: "PATUH",
    VIOLATION: "MELANGGAR",
    VIOLATION_INFERRED: "DUGAAN LANGGAR",
    NEEDS_REVIEW: "PERLU TINJAU",
}

PPE_COLOR = (148, 163, 184)  # abu-abu; APD sekadar bukti pendukung


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except Exception:  # noqa: BLE001
            continue
    return ImageFont.load_default()


def _label_box(draw: ImageDraw.ImageDraw, xy: Tuple[float, float],
               text: str, color: Tuple[int, int, int],
               font: ImageFont.ImageFont) -> None:
    """Teks berlatar solid supaya terbaca di atas gambar apa pun."""
    x, y = xy
    try:
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        tw, th = r - l, b - t
    except Exception:  # noqa: BLE001
        tw, th = len(text) * 7, 12

    pad = 3
    y = max(0, y - th - 2 * pad)
    draw.rectangle([x, y, x + tw + 2 * pad, y + th + 2 * pad], fill=color)
    draw.text((x + pad, y + pad), text, fill=(255, 255, 255), font=font)


def annotate(image, report: Dict[str, object],
             show_ppe: bool = True,
             show_legend: bool = True) -> Image.Image:
    """
    Gambar hasil penilaian di atas citra asli.

    `report` adalah keluaran pipeline.analyze(), yang masih memuat kunci
    internal `_workers` dan `_orphans`.
    """
    if isinstance(image, Image.Image):
        img = image.convert("RGB").copy()
    elif isinstance(image, np.ndarray):
        img = Image.fromarray(image).convert("RGB")
    else:
        img = Image.open(image).convert("RGB")

    draw = ImageDraw.Draw(img)
    scale = max(img.width, img.height) / 1000
    lw = max(2, int(3 * scale))
    font = _font(max(12, int(16 * scale)))
    small = _font(max(10, int(13 * scale)))

    workers = report.get("_workers", [])
    infos = report.get("workers", [])

    # APD lebih dulu supaya kotak pekerja berada di lapisan atas
    if show_ppe:
        for w in workers:
            for det in (w.head, w.torso):
                if det is None:
                    continue
                draw.rectangle(list(det.xyxy), outline=PPE_COLOR,
                               width=max(1, lw - 1))
                _label_box(draw, (det.xyxy[0], det.xyxy[1]),
                           f"{det.class_name} {det.confidence:.2f}",
                           PPE_COLOR, small)

        for o in report.get("_orphans", []):
            draw.rectangle(list(o.xyxy), outline=(168, 85, 247), width=lw)
            _label_box(draw, (o.xyxy[0], o.xyxy[1]),
                       f"{o.class_name} (tanpa pemilik)", (168, 85, 247), small)

    for info in infos:
        color = STATUS_COLOR.get(info["overall"], (100, 100, 100))
        x1, y1, x2, y2 = info["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=lw + 1)
        _label_box(draw, (x1, y1),
                   f"#{info['index']} {STATUS_LABEL.get(info['overall'], info['overall'])}",
                   color, font)

    if show_legend:
        _draw_summary(draw, report, img.width, small)

    return img


def _draw_summary(draw: ImageDraw.ImageDraw, report: Dict[str, object],
                  width: int, font: ImageFont.ImageFont) -> None:
    """Panel ringkas di pojok kiri atas."""
    s = report.get("summary", {})
    lines = [
        f"Pekerja: {s.get('n_workers', 0)}",
        f"Patuh: {s.get('n_compliant', 0)}  "
        f"Melanggar: {s.get('n_violation', 0)}  "
        f"Perlu tinjau: {s.get('n_needs_review', 0)}",
    ]
    if s.get("assessable", 0):
        lines.append(f"Tingkat kepatuhan: {s['compliance_rate']:.0%} "
                     f"(dari {s['assessable']} yang dapat dinilai)")

    pad = 8
    lh = 16
    box_h = pad * 2 + lh * len(lines)
    draw.rectangle([0, 0, min(width, 420), box_h], fill=(15, 23, 42))
    for i, line in enumerate(lines):
        draw.text((pad, pad + i * lh), line, fill=(226, 232, 240), font=font)
