#!/usr/bin/env python3
"""
Bangkitkan train set yang diperluas dengan augmentasi fotometrik.

Contoh:
    python run_augment.py --dataset-root /content/dataset \
        --output-root /content/dataset_aug \
        --variants 1 --variants-minority 4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from training.augment_offline import generate  # noqa: E402
from training.config import MINORITY_CLASSES   # noqa: E402


def main() -> int:
    """
    Entry point CLI: baca argumen, tentukan kelas minoritas dari data.yaml,
    lalu panggil `generate()` untuk menulis varian augmentasi ke disk.
    """
    p = argparse.ArgumentParser(description="Augmentasi fotometrik offline")
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--variants", type=int, default=1,
                   help="Varian per gambar biasa")
    p.add_argument("--variants-minority", type=int, default=4,
                   help="Varian per gambar yang memuat kelas minoritas")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)-8s %(name)-24s %(message)s")

    import yaml
    data = yaml.safe_load((Path(args.dataset_root) / "data.yaml")
                          .read_text(encoding="utf-8"))
    names = data.get("names", [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    minority_ids = {names.index(c) for c in MINORITY_CLASSES if c in names}
    print(f"Kelas minoritas: {[names[i] for i in minority_ids]} -> id {minority_ids}")

    summary = generate(
        dataset_root=Path(args.dataset_root),
        output_root=Path(args.output_root),
        minority_class_ids=minority_ids,
        n_variants=args.variants,
        n_variants_minority=args.variants_minority,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
