"""
Konfigurasi inferensi kepatuhan APD.

Seluruh angka di sini BUKAN tebakan. Threshold per kelas diturunkan dari
sapuan kurva F2/F1 pada run terpilih (lihat evaluation_test.json), dan ambang
containment 0.6 diturunkan dari kurva sensitivitas pada EDA.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

PERSON = "person"
HEAD_POSITIVE, HEAD_NEGATIVE = "helmet", "no-helmet"
TORSO_POSITIVE, TORSO_NEGATIVE = "vest", "no-vest"

HEAD_CLASSES = (HEAD_POSITIVE, HEAD_NEGATIVE)
TORSO_CLASSES = (TORSO_POSITIVE, TORSO_NEGATIVE)
PPE_CLASSES = HEAD_CLASSES + TORSO_CLASSES

# Status per bagian tubuh
COMPLIANT = "COMPLIANT"
VIOLATION = "VIOLATION"            # terdeteksi eksplisit (jalur A)
VIOLATION_INFERRED = "VIOLATION*"  # disimpulkan dari ketiadaan (jalur B)
NEEDS_REVIEW = "NEEDS_REVIEW"      # tidak dapat dipastikan

# Threshold hasil sapuan F2 (kelas pelanggaran) / F1 (kelas lain)
# pada run imgsz_960. Dipakai bila evaluation JSON tidak tersedia.
DEFAULT_THRESHOLDS: Dict[str, float] = {
    "person": 0.50,
    "helmet": 0.45,
    "no-helmet": 0.20,
    "vest": 0.15,
    "no-vest": 0.20,
}


@dataclass
class InferenceConfig:
    weights: Path
    thresholds: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_THRESHOLDS))

    imgsz: int = 960
    device: str = "cpu"

    # Ambang containment: area(irisan) / area(box APD).
    # 0.6 dipilih dari kurva sensitivitas EDA - attach rate 95.8%
    # dengan ambiguity hanya 0.6%.
    containment_threshold: float = 0.60

    # Zona anatomis (fraksi tinggi bbox person, dari atas).
    # Divalidasi pada EDA: 99.7% helmet dan 96.3% vest berada di dalamnya.
    # Dipakai sebagai PENAMBAH SKOR, bukan filter keras - 3.7% vest berada
    # di luar zona, terutama pada postur jongkok/membungkuk.
    head_zone: tuple = (0.00, 0.40)
    torso_zone: tuple = (0.20, 0.75)
    zone_bonus: float = 0.15

    # Ambang untuk memutuskan VIOLATION* vs NEEDS_REVIEW ketika tidak ada
    # APD yang terasosiasi pada suatu bagian tubuh.
    edge_margin: float = 0.02        # fraksi lebar/tinggi gambar
    occlusion_iou: float = 0.35      # tumpang tindih dengan person lain

    # Mode jalur B (pelanggaran yang disimpulkan dari ketiadaan APD).
    #   "violation" - vonis pelanggaran penuh
    #   "review"    - ditandai untuk ditinjau manusia  <-- default
    #   "off"       - abaikan sama sekali
    #
    # Default "review" ditetapkan dari hasil validasi terhadap ground truth
    # pada test set: presisi jalur B hanya 0.31 (8 benar dari 26 tuduhan).
    # 18 pekerja yang sebenarnya patuh akan dituduh melanggar - sebagian
    # besar karena model gagal mendeteksi APD yang sebenarnya ada, bukan
    # karena APD-nya memang tidak ada.
    path_b_mode: str = "review"

    def __post_init__(self) -> None:
        self.weights = Path(self.weights)

    # ------------------------------------------------------------------
    @classmethod
    def from_evaluation(cls, weights: Path, eval_json: Path,
                        **kwargs) -> "InferenceConfig":
        """
        Bangun konfigurasi dengan threshold yang diambil dari hasil evaluasi.

        Ini menutup celah yang lazim terjadi: model dilatih dan dievaluasi
        dengan threshold hasil optimasi, lalu di-deploy dengan conf=0.25
        bawaan. Threshold adalah bagian dari model, bukan detail teknis.
        """
        cfg = cls(weights=weights, **kwargs)
        path = Path(eval_json)
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            per_class = data.get("per_class", {})
            for name, info in per_class.items():
                thr = info.get("recommended_threshold")
                if thr is not None:
                    cfg.thresholds[name] = float(thr)
        return cfg

    def threshold(self, class_name: str) -> float:
        return self.thresholds.get(class_name, 0.25)

    @property
    def min_threshold(self) -> float:
        """Threshold terendah - dipakai sebagai conf saat memanggil model."""
        return min(self.thresholds.values())

    def to_dict(self) -> dict:
        return {
            "weights": str(self.weights),
            "imgsz": self.imgsz,
            "thresholds": self.thresholds,
            "containment_threshold": self.containment_threshold,
            "zone_bonus": self.zone_bonus,
        }
