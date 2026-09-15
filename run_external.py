#!/usr/bin/env python3
"""
Alat bantu dataset eksternal: audit kebocoran, inspeksi sumber, penggabungan.

Alur yang disarankan:

    # 1. Audit dataset yang ada - apakah sudah ada kebocoran nyaris-duplikat?
    python run_external.py audit --dataset-root /content/dataset

    # 2. Periksa taksonomi sumber eksternal SEBELUM memutuskan apa pun
    python run_external.py inspect --external-root /content/ext_hardhat

    # 3. Gabungkan, dengan pemetaan kelas eksplisit
    python run_external.py merge \
        --external-root /content/ext_hardhat \
        --target-root /content/dataset_plus \
        --protect-dataset /content/dataset \
        --map "Hardhat=helmet,NO-Hardhat=no-helmet,Person=person,Safety Vest=vest,NO-Safety Vest=no-vest" \
        --want no-helmet --max-images 400
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from training.external_data import (  # noqa: E402
    find_near_duplicates, inspect_external, merge_external,
)


def parse_map(text: str) -> dict:
    out = {}
    for pair in text.split(","):
        if not pair.strip():
            continue
        k, _, v = pair.partition("=")
        out[k.strip()] = None if v.strip().lower() in ("", "none", "drop") else v.strip()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Alat bantu dataset eksternal")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("audit", help="Cari kebocoran nyaris-duplikat lintas split")
    a.add_argument("--dataset-root", required=True)
    a.add_argument("--threshold", type=int, default=6)

    i = sub.add_parser("inspect", help="Periksa taksonomi sumber eksternal")
    i.add_argument("--external-root", required=True)

    m = sub.add_parser("merge", help="Gabungkan ke split train")
    m.add_argument("--external-root", required=True)
    m.add_argument("--target-root", required=True)
    m.add_argument("--protect-dataset", required=True,
                   help="Akar dataset proyek; valid/test dipakai sebagai acuan anti-bocor")
    m.add_argument("--map", required=True, help="nama_eksternal=nama_proyek, dipisah koma")
    m.add_argument("--want", default="no-helmet", help="kelas target, dipisah koma")
    m.add_argument("--max-images", type=int, default=400)
    m.add_argument("--dup-threshold", type=int, default=8)
    m.add_argument("--copy-base", action="store_true", default=True,
                   help="Salin dataset proyek ke target lebih dulu")

    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")

    if args.cmd == "audit":
        print(json.dumps(find_near_duplicates(Path(args.dataset_root),
                                              threshold=args.threshold),
                         indent=2, ensure_ascii=False))
        return 0

    if args.cmd == "inspect":
        print(json.dumps(inspect_external(Path(args.external_root)),
                         indent=2, ensure_ascii=False))
        return 0

    # merge
    import yaml
    base = Path(args.protect_dataset)
    tgt = Path(args.target_root)

    if args.copy_base and not tgt.exists():
        print(f"Menyalin dataset dasar ke {tgt} ...")
        shutil.copytree(base, tgt)

    names = yaml.safe_load((base / "data.yaml").read_text(encoding="utf-8")).get("names", [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]

    res = merge_external(
        external_root=Path(args.external_root),
        target_root=tgt,
        class_map=parse_map(args.map),
        target_class_names=names,
        want_classes=tuple(w.strip() for w in args.want.split(",")),
        protect_dataset=base,
        max_images=args.max_images,
        dup_threshold=args.dup_threshold,
    )
    print(json.dumps(res, indent=2, ensure_ascii=False))

    if res.get("accepted"):
        data = yaml.safe_load((tgt / "data.yaml").read_text(encoding="utf-8"))
        data["train"] = str((tgt / "train" / "images").resolve())
        data["val"] = str((tgt / "valid" / "images").resolve())
        data["test"] = str((tgt / "test" / "images").resolve())
        (tgt / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False),
                                       encoding="utf-8")
        print(f"\ndata.yaml diperbarui: {tgt / 'data.yaml'}")
    return 0 if res.get("accepted") else 1


if __name__ == "__main__":
    raise SystemExit(main())
