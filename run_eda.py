#!/usr/bin/env python3
"""
Orchestrator EDA - satu titik masuk untuk seluruh pipeline.

Contoh pemakaian:
    python run_eda.py --dataset-root ./construction-safety --output-dir eda_output
    python run_eda.py --dataset-root /content --containment-threshold 0.5

Urutan eksekusi disengaja:
    load -> integrity -> distribution -> geometry -> association -> visualize -> report

Integritas dijalankan lebih dulu karena statistik di atas data kotor
menghasilkan kesimpulan yang salah namun terlihat meyakinkan.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from eda import EDAConfig
from eda import association, distribution, geometry, integrity, loader, report, visualize


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EDA dataset Construction Safety (format YOLO)")
    p.add_argument("--dataset-root", required=True,
                   help="Folder yang memuat data.yaml serta train/valid/test")
    p.add_argument("--output-dir", default="eda_output")
    p.add_argument("--containment-threshold", type=float, default=0.60)
    p.add_argument("--sample-images", type=int, default=12)
    p.add_argument("--skip-figures", action="store_true",
                   help="Lewati pembuatan figur (mempercepat iterasi)")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-8s %(name)-18s %(message)s",
    )
    log = logging.getLogger("run_eda")

    cfg = EDAConfig(
        dataset_root=Path(args.dataset_root),
        output_dir=Path(args.output_dir),
        containment_threshold=args.containment_threshold,
        sample_images=args.sample_images,
    )
    log.info("Kelas terbaca: %s", cfg.class_names)
    log.info("Split tersedia: %s", cfg.available_splits())

    log.info("[1/6] Memuat anotasi ...")
    images_df, boxes_df, errors = loader.load_dataset(cfg)
    if images_df.empty:
        log.error("Tidak ada gambar ditemukan. Periksa --dataset-root.")
        return 1

    log.info("[2/6] Memeriksa integritas ...")
    integ = integrity.run(cfg, images_df, boxes_df, errors)

    log.info("[3/6] Menganalisis sebaran kelas ...")
    dist = distribution.run(cfg, images_df, boxes_df)

    log.info("[4/6] Menganalisis geometri objek ...")
    geom = geometry.run(cfg, images_df, boxes_df)

    log.info("[5/6] Menguji kelayakan asosiasi person-APD ...")
    assoc = association.run(cfg, boxes_df)

    figures = {}
    if not args.skip_figures:
        log.info("[6/6] Membuat figur ...")
        figures = visualize.run(cfg, images_df, boxes_df, dist, geom, assoc)

    paths = report.run(cfg, integ, dist, geom, assoc, figures)

    print("\n" + "=" * 62)
    print("EDA SELESAI")
    print("=" * 62)
    print(f"Gambar          : {integ['n_images']}")
    print(f"Bounding box    : {integ['n_boxes']}")
    print(f"Duplikat leakage: {integ['n_cross_split_duplicates']}")
    if "attach_rate" in assoc:
        print(f"Person coverage : {assoc['person_coverage']:.1%}")
        print(f"Attach rate     : {assoc['attach_rate']:.1%}")
        print(f"Ambiguity rate  : {assoc['ambiguity_rate']:.1%}")
    print(f"\nVERDICT: {assoc.get('verdict', assoc.get('reason', '-'))}")
    print(f"\nLaporan : {paths['markdown']}")
    print(f"JSON    : {paths['json']}")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
