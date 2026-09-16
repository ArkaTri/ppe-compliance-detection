"""
Konfigurasi training.

Seluruh hyperparameter ditulis EKSPLISIT, termasuk yang nilainya sama dengan
default Ultralytics. Alasannya: default augmentasi Ultralytics mengaktifkan
`fliplr=0.5` dan `mosaic=1.0` (keduanya geometris) sehingga panduan capstone
dilanggar tanpa disadari jika parameter dibiarkan implisit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

# Kelas yang jarang muncul dan kritis secara keselamatan.
# Berdasarkan EDA: no-helmet hanya 129 instance (ambang minimum 200).
MINORITY_CLASSES: tuple = ("no-helmet",)

# Kelas pelanggaran. Recall lebih penting daripada precision di sini:
# melewatkan pelanggaran (false negative) lebih mahal daripada alarm palsu.
VIOLATION_CLASSES: tuple = ("no-helmet", "no-vest")


# --------------------------------------------------------------------------
# Kebijakan augmentasi
# --------------------------------------------------------------------------
# COMPLIANT  : sesuai panduan capstone - hanya augmentasi fotometrik.
# MOSAIC_ABL : untuk ablation study, mengukur BERAPA HARGA kepatuhan tersebut.
#              Bukan untuk model final.

AUG_COMPLIANT: Dict[str, float] = {
    # -- fotometrik (diizinkan) --
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    # -- geometris (dimatikan eksplisit) --
    "degrees": 0.0,
    "translate": 0.0,
    "scale": 0.0,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.0,
    "bgr": 0.0,
    # -- compositing (geometris secara efektif) --
    "mosaic": 0.0,
    "mixup": 0.0,
    "cutmix": 0.0,
    "copy_paste": 0.0,
}

AUG_MOSAIC_ABLATION: Dict[str, float] = {
    **AUG_COMPLIANT,
    "mosaic": 1.0,
    "close_mosaic": 10,  # matikan mosaic pada 10 epoch terakhir
}

AUG_POLICIES: Dict[str, Dict[str, float]] = {
    "compliant": AUG_COMPLIANT,
    "mosaic_ablation": AUG_MOSAIC_ABLATION,
}


@dataclass
class TrainConfig:
    """Parameter satu run training."""

    dataset_root: Path
    run_name: str = "baseline_640"

    # -- model & komputasi ------------------------------------------------
    weights: str = "yolo12n.pt"
    epochs: int = 100
    batch: int = 16
    imgsz: int = 640
    patience: int = 20
    device: str = ""          # "" = auto, "0" = GPU pertama, "cpu"
    workers: int = 8
    seed: int = 42

    # -- kebijakan --------------------------------------------------------
    aug_policy: str = "compliant"
    oversample_factor: int = 3   # 1 = nonaktif

    # -- output -----------------------------------------------------------
    project_dir: Path = Path("training_output")

    # -- diisi otomatis ---------------------------------------------------
    class_names: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """
    Validasi aug_policy yang dipilih, siapkan folder project, dan muat nama
    kelas dari data.yaml jika belum diisi.
    """
        self.dataset_root = Path(self.dataset_root)
        self.project_dir = Path(self.project_dir)
        self.project_dir.mkdir(parents=True, exist_ok=True)

        if self.aug_policy not in AUG_POLICIES:
            raise ValueError(
                f"aug_policy '{self.aug_policy}' tidak dikenal. "
                f"Pilihan: {list(AUG_POLICIES)}"
            )

        if not self.class_names:
            import yaml
            yaml_path = self.dataset_root / "data.yaml"
            if not yaml_path.exists():
                raise FileNotFoundError(f"data.yaml tidak ditemukan di {yaml_path}")
            data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
            names = data.get("names", [])
            if isinstance(names, dict):
                names = [names[k] for k in sorted(names)]
            self.class_names = list(names)

    # -- helper -----------------------------------------------------------
    @property
    def augmentation(self) -> Dict[str, float]:
        """Ambil dict parameter augmentasi sesuai aug_policy yang dipilih."""
        return dict(AUG_POLICIES[self.aug_policy])

    @property
    # run_dir: folder output khusus run ini. best_weights: path bobot terbaik.
    # class_id: konversi nama kelas -> index integer sesuai urutan di data.yaml.
    def run_dir(self) -> Path:
        return self.project_dir / self.run_name

    @property
    # run_dir: folder output khusus run ini. best_weights: path bobot terbaik.
    # class_id: konversi nama kelas -> index integer sesuai urutan di data.yaml.
    def best_weights(self) -> Path:
        return self.run_dir / "weights" / "best.pt"

    def class_id(self, name: str) -> int:
    # run_dir: folder output khusus run ini. best_weights: path bobot terbaik.
    # class_id: konversi nama kelas -> index integer sesuai urutan di data.yaml.
        return self.class_names.index(name)

    # Ringkasan seluruh parameter training — disimpan sebagai bukti provenance run.
    def to_dict(self) -> dict:
        return {
            "run_name": self.run_name,
            "weights": self.weights,
            "epochs": self.epochs,
            "batch": self.batch,
            "imgsz": self.imgsz,
            "patience": self.patience,
            "seed": self.seed,
            "aug_policy": self.aug_policy,
            "oversample_factor": self.oversample_factor,
            "augmentation": self.augmentation,
        }
