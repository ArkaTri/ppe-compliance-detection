"""
Ablation: menjalankan beberapa konfigurasi dan membandingkannya secara adil.

Tiga run yang dirancang, masing-masing menjawab satu pertanyaan:

  baseline_640      - model final. Patuh panduan, oversampling aktif.
  imgsz_960         - apakah resolusi lebih tinggi menolong `no-helmet`,
                      kelas yang paling langka SEKALIGUS paling kecil?
  mosaic_ablation   - berapa harga kepatuhan terhadap larangan augmentasi
                      geometris? Bukan kandidat model final, tetapi angkanya
                      membuat keputusan menjadi berdasar, bukan sekadar taat.

Perbandingan difokuskan pada recall per kelas dengan interval kepercayaan,
bukan pada mAP agregat. Dari EDA: rasio ketidakseimbangan 21.8x membuat mAP
agregat didominasi kelas mayoritas.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

from .config import TrainConfig, VIOLATION_CLASSES
from . import evaluate, train as train_mod

logger = logging.getLogger(__name__)


def run_matrix(dataset_root: Path, project_dir: Path,
               epochs: int, batch: int, device: str) -> List[TrainConfig]:
    """Definisi tiga run ablation."""
    common = dict(dataset_root=dataset_root, project_dir=project_dir,
                  epochs=epochs, batch=batch, device=device)
    return [
        TrainConfig(run_name="baseline_640", imgsz=640,
                    aug_policy="compliant", oversample_factor=3, **common),
        TrainConfig(run_name="imgsz_960", imgsz=960,
                    aug_policy="compliant", oversample_factor=3, **common),
        TrainConfig(run_name="mosaic_ablation", imgsz=640,
                    aug_policy="mosaic_ablation", oversample_factor=3, **common),
    ]


def evaluate_run(cfg: TrainConfig, split: str = "test",
                 n_boot: int = 1000) -> Dict[str, object]:
    """Evaluasi satu run: threshold optimal + recall dengan CI per kelas."""
    from ultralytics import YOLO

    if not cfg.best_weights.exists():
        raise FileNotFoundError(f"Bobot tidak ditemukan: {cfg.best_weights}")

    model = YOLO(str(cfg.best_weights))
    per_image = evaluate.collect_matches(model, cfg.dataset_root, split)

    per_class: Dict[str, object] = {}
    for name in cfg.class_names:
        cid = cfg.class_id(name)
        is_violation = name in VIOLATION_CLASSES
        rec = evaluate.recommend_threshold(per_image, cid, is_violation)
        boot = evaluate.bootstrap_metric(per_image, cid, rec["threshold"],
                                         metric="recall", n_boot=n_boot,
                                         seed=cfg.seed)
        per_class[name] = {
            "recommended_threshold": rec["threshold"],
            "optimized_for": rec["optimized_for"],
            "precision": rec["precision"],
            "recall": rec["recall"],
            "f1": rec["f1"],
            "f2": rec["f2"],
            "n_gt": rec["n_gt"],
            "recall_ci95": [boot.get("ci95_low"), boot.get("ci95_high")],
            "ci_width": boot.get("ci_width"),
        }

    out = {
        "run_name": cfg.run_name,
        "split": split,
        "imgsz": cfg.imgsz,
        "aug_policy": cfg.aug_policy,
        "per_class": per_class,
    }
    (cfg.run_dir / f"evaluation_{split}.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return out


def comparison_markdown(results: List[Dict[str, object]],
                        class_names: List[str]) -> str:
    """Tabel perbandingan antar run, satu tabel per kelas."""
    L: List[str] = []
    A = L.append

    A("# Perbandingan Ablation")
    A("")
    A("Metrik utama adalah **recall per kelas** dengan interval kepercayaan 95% "
      "(bootstrap atas gambar). mAP agregat sengaja tidak dijadikan kriteria "
      "utama karena rasio ketidakseimbangan 21.8x membuatnya didominasi kelas "
      "mayoritas.")
    A("")

    for name in class_names:
        A(f"## `{name}`")
        A("")
        A("| Run | imgsz | Augmentasi | Thr | Precision | Recall | CI 95% | n GT |")
        A("|---|---|---|---|---|---|---|---|")
        for r in results:
            pc = r["per_class"].get(name)
            if not pc:
                continue
            lo, hi = pc["recall_ci95"]
            ci = f"[{lo}, {hi}]" if lo is not None else "-"
            A(f"| {r['run_name']} | {r['imgsz']} | {r['aug_policy']} | "
              f"{pc['recommended_threshold']} | {pc['precision']} | "
              f"{pc['recall']} | {ci} | {pc['n_gt']} |")
        A("")

    A("## Cara membaca tabel ini")
    A("")
    A("- **Interval yang saling tumpang tindih berarti selisihnya tidak "
      "signifikan.** Jangan mengklaim satu konfigurasi lebih baik jika CI-nya "
      "beririsan; pada kelas dengan n GT kecil hal ini sangat mungkin terjadi.")
    A("- **Threshold berbeda antar kelas adalah keputusan yang disengaja.** "
      "Kelas pelanggaran dioptimalkan pada F2 (recall dibobot dua kali), kelas "
      "lain pada F1, karena melewatkan pelanggaran lebih mahal daripada alarm "
      "palsu di konteks keselamatan kerja.")
    A("- **`mosaic_ablation` bukan kandidat model final.** Run ini hanya "
      "mengukur berapa harga kepatuhan terhadap larangan augmentasi geometris.")
    A("")
    return "\n".join(L)
