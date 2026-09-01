#!/usr/bin/env python3
"""
Orchestrator training & ablation.

Contoh:
    # satu run saja (model final)
    python run_training.py --dataset-root ./dataset --runs baseline_640

    # seluruh matriks ablation
    python run_training.py --dataset-root ./dataset --runs all --epochs 100

    # hanya evaluasi ulang dari bobot yang sudah ada
    python run_training.py --dataset-root ./dataset --runs all --eval-only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from training import ablation, train as train_mod  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Training & ablation PPE detection")
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--project-dir", default="training_output")
    p.add_argument("--runs", default="baseline_640",
                   help="'all' atau nama run dipisah koma")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="")
    p.add_argument("--eval-split", default="test")
    p.add_argument("--eval-only", action="store_true",
                   help="Lewati training, evaluasi bobot yang sudah ada")
    p.add_argument("--n-bootstrap", type=int, default=1000)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)-8s %(name)-20s %(message)s")
    log = logging.getLogger("run_training")

    project_dir = Path(args.project_dir)
    all_cfgs = ablation.run_matrix(
        dataset_root=Path(args.dataset_root),
        project_dir=project_dir,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
    )

    if args.runs.strip().lower() == "all":
        selected = all_cfgs
    else:
        wanted = {r.strip() for r in args.runs.split(",")}
        selected = [c for c in all_cfgs if c.run_name in wanted]
        unknown = wanted - {c.run_name for c in all_cfgs}
        if unknown:
            log.error("Run tidak dikenal: %s. Tersedia: %s",
                      unknown, [c.run_name for c in all_cfgs])
            return 1

    results = []
    for cfg in selected:
        if not args.eval_only:
            log.info("=" * 60)
            log.info("TRAINING: %s", cfg.run_name)
            log.info("=" * 60)
            train_mod.train(cfg)

        log.info("Evaluasi '%s' pada split '%s' ...", cfg.run_name, args.eval_split)
        results.append(ablation.evaluate_run(cfg, split=args.eval_split,
                                             n_boot=args.n_bootstrap))

    md = ablation.comparison_markdown(results, selected[0].class_names)
    md_path = project_dir / "ablation_report.md"
    md_path.write_text(md, encoding="utf-8")
    (project_dir / "ablation_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 60)
    print("SELESAI")
    print("=" * 60)
    for r in results:
        print(f"\n[{r['run_name']}] imgsz={r['imgsz']} aug={r['aug_policy']}")
        for name, pc in r["per_class"].items():
            lo, hi = pc["recall_ci95"]
            ci = f"[{lo}, {hi}]" if lo is not None else "-"
            print(f"  {name:<12} thr={pc['recommended_threshold']:<5} "
                  f"R={pc['recall']:<7} CI={ci:<18} n={pc['n_gt']}")
    print(f"\nLaporan: {md_path}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
