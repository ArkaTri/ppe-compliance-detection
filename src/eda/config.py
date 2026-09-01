"""
Konfigurasi terpusat untuk pipeline EDA.

Semua konstanta yang bisa berubah (path, nama kelas, ambang batas) dikumpulkan
di sini supaya modul lain tidak perlu diubah saat dataset atau lingkungan
berpindah (lokal <-> Google Colab).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import yaml

# --------------------------------------------------------------------------
# Semantik kelas (khusus dataset Construction Safety)
# --------------------------------------------------------------------------
# Pemisahan ini penting: modul association perlu tahu mana "wadah" (person)
# dan mana "atribut" (APD), serta di bagian tubuh mana APD seharusnya berada.

PERSON_CLASS = "person"

HEAD_CLASSES = ("helmet", "no-helmet")
TORSO_CLASSES = ("vest", "no-vest")

PPE_CLASSES = HEAD_CLASSES + TORSO_CLASSES

# Kelas yang menandakan pelanggaran. Dipakai untuk analisis kelas minoritas.
VIOLATION_CLASSES = ("no-helmet", "no-vest")

# Zona vertikal yang diharapkan (fraksi tinggi bbox person, diukur dari atas).
# Dipakai sebagai hipotesis yang DIUJI oleh modul association, bukan sebagai
# aturan yang diasumsikan benar.
EXPECTED_ZONES: Dict[str, tuple] = {
    "head": (0.00, 0.40),
    "torso": (0.20, 0.75),
}


@dataclass
class EDAConfig:
    """Parameter runtime untuk seluruh pipeline EDA."""

    # -- path -------------------------------------------------------------
    dataset_root: Path
    output_dir: Path = Path("eda_output")
    splits: List[str] = field(default_factory=lambda: ["train", "valid", "test"])

    # -- ambang analisis --------------------------------------------------
    # Rasio containment minimum agar sebuah box APD dianggap "milik" seorang
    # person: area(irisan) / area(box APD).
    containment_threshold: float = 0.60

    # Ambang luas relatif (fraksi luas gambar) untuk klasifikasi ukuran objek.
    # Mengikuti semangat definisi COCO small/medium/large, dinormalisasi.
    small_area_frac: float = 0.0032   # ~32x32 px pada gambar 640x640
    medium_area_frac: float = 0.0290  # ~96x96 px pada gambar 640x640

    # Batas keputusan (dipakai modul report untuk verdict GO / NO-GO).
    min_violation_instances: int = 200
    min_person_coverage: float = 0.70   # fraksi gambar ber-APD yang punya person
    min_ppe_attach_rate: float = 0.70   # fraksi box APD yang menemukan person

    # -- visual -----------------------------------------------------------
    sample_images: int = 12
    random_seed: int = 42

    # -- diisi otomatis ---------------------------------------------------
    class_names: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.dataset_root = Path(self.dataset_root)
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "figures").mkdir(exist_ok=True)
        (self.output_dir / "samples").mkdir(exist_ok=True)

        if not self.class_names:
            self.class_names = self._load_class_names()

    def _load_class_names(self) -> List[str]:
        yaml_path = self.dataset_root / "data.yaml"
        if not yaml_path.exists():
            raise FileNotFoundError(
                f"data.yaml tidak ditemukan di {yaml_path}. "
                "Pastikan --dataset-root menunjuk ke folder hasil ekstrak dataset."
            )
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        names = data.get("names", [])
        if isinstance(names, dict):  # format {0: 'helmet', ...}
            names = [names[k] for k in sorted(names)]
        return list(names)

    # -- helper path ------------------------------------------------------
    def images_dir(self, split: str) -> Path:
        return self.dataset_root / split / "images"

    def labels_dir(self, split: str) -> Path:
        return self.dataset_root / split / "labels"

    def available_splits(self) -> List[str]:
        """Hanya kembalikan split yang benar-benar ada di disk."""
        return [s for s in self.splits if self.images_dir(s).is_dir()]
