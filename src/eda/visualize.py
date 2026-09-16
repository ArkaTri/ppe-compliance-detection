"""
Visualize: menghasilkan figur untuk laporan dan video penjelasan.

Semua figur disimpan ke <output_dir>/figures sebagai PNG agar bisa langsung
disisipkan ke README GitHub, slide, atau rekaman video.

Modul ini sengaja dipisah dari modul analisis: perubahan gaya visual tidak
boleh memaksa perhitungan diulang.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")  # backend non-interaktif, aman di Colab & server
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from .config import EDAConfig, PERSON_CLASS, PPE_CLASSES

logger = logging.getLogger(__name__)

PALETTE = {
    "person": "#3b82f6",
    "helmet": "#22c55e",
    "vest": "#14b8a6",
    "no-helmet": "#ef4444",
    "no-vest": "#f97316",
}


def _save(fig: plt.Figure, path: Path) -> str:
    """Helper bersama: simpan figure matplotlib ke PNG lalu tutup memori-nya."""
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figur disimpan: %s", path)
    return str(path)


def plot_class_distribution(counts: pd.DataFrame, cfg: EDAConfig) -> str:
    """Gambar satu figur EDA dan simpan ke folder figures/ untuk laporan."""
    split_cols = [c for c in counts.columns if c != "TOTAL"]
    fig, ax = plt.subplots(figsize=(9, 5))
    counts[split_cols].plot(kind="barh", stacked=True, ax=ax,
                            color=["#1e3a8a", "#60a5fa", "#bfdbfe"][:len(split_cols)])
    ax.set_xlabel("Jumlah instance")
    ax.set_ylabel("")
    ax.set_title("Sebaran instance per kelas dan split")
    ax.grid(axis="x", alpha=0.25)
    return _save(fig, cfg.output_dir / "figures" / "class_distribution.png")


def plot_size_distribution(size_dist: pd.DataFrame, cfg: EDAConfig) -> str:
    """Gambar satu figur EDA dan simpan ke folder figures/ untuk laporan."""
    fig, ax = plt.subplots(figsize=(9, 5))
    cols = [c for c in ["small", "medium", "large"] if c in size_dist.columns]
    size_dist[cols].plot(kind="barh", stacked=True, ax=ax,
                         color=["#dc2626", "#f59e0b", "#16a34a"][:len(cols)])
    ax.set_xlabel("Proporsi")
    ax.set_title("Proporsi ukuran objek per kelas (small / medium / large)")
    ax.legend(title="Kategori", loc="lower right")
    return _save(fig, cfg.output_dir / "figures" / "size_distribution.png")


def plot_area_boxplot(boxes_df: pd.DataFrame, cfg: EDAConfig) -> str:
    """Gambar satu figur EDA dan simpan ke folder figures/ untuk laporan."""
    fig, ax = plt.subplots(figsize=(9, 5))
    order = boxes_df.groupby("class_name")["area_frac"].median().sort_values().index
    data = [boxes_df.loc[boxes_df["class_name"] == c, "area_frac"] for c in order]
    # matplotlib >= 3.9 mengganti `labels` menjadi `tick_labels`, dan
    # >= 3.10 mengganti `vert=False` menjadi `orientation="horizontal"`.
    # Argumen lama dihapus pada versi terbaru, jadi dicoba berurutan.
    kw = dict(showfliers=False)
    for extra in (
        {"tick_labels": list(order), "orientation": "horizontal"},
        {"tick_labels": list(order), "vert": False},
        {"labels": list(order), "vert": False},
    ):
        try:
            ax.boxplot(data, **kw, **extra)
            break
        except TypeError:
            continue
    else:
        ax.boxplot(data, **kw)
        ax.set_yticklabels(list(order))
    ax.set_xscale("log")
    ax.set_xlabel("Luas box relatif terhadap luas gambar (skala log)")
    ax.set_title("Sebaran ukuran bounding box per kelas")
    ax.grid(axis="x", alpha=0.25)
    return _save(fig, cfg.output_dir / "figures" / "area_boxplot.png")


def plot_association_sensitivity(sensitivity: pd.DataFrame, cfg: EDAConfig) -> str:
    """Gambar satu figur EDA dan simpan ke folder figures/ untuk laporan."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sensitivity["threshold"], sensitivity["attach_rate"],
            marker="o", label="Attach rate", color="#16a34a")
    ax.plot(sensitivity["threshold"], sensitivity["ambiguity_rate"],
            marker="s", label="Ambiguity rate", color="#dc2626")
    ax.axvline(cfg.containment_threshold, ls="--", color="#64748b",
               label=f"Ambang terpilih = {cfg.containment_threshold}")
    ax.set_xlabel("Ambang containment ratio")
    ax.set_ylabel("Proporsi box APD")
    ax.set_title("Sensitivitas asosiasi APD terhadap ambang containment")
    ax.legend()
    ax.grid(alpha=0.25)
    return _save(fig, cfg.output_dir / "figures" / "association_sensitivity.png")


def plot_zone_positions(assoc_df: pd.DataFrame, cfg: EDAConfig) -> str:
    """Histogram posisi vertikal relatif APD di dalam bbox person."""
    matched = assoc_df[assoc_df["matched"]]
    classes = [c for c in PPE_CLASSES if c in set(matched["ppe_class"])]
    if not classes:
        return ""
    fig, axes = plt.subplots(1, len(classes), figsize=(4 * len(classes), 3.6),
                             sharey=True)
    axes = [axes] if len(classes) == 1 else list(axes)
    for ax, cls in zip(axes, classes):
        vals = matched.loc[matched["ppe_class"] == cls, "rel_y"].dropna()
        ax.hist(vals, bins=25, range=(0, 1), color=PALETTE.get(cls, "#64748b"))
        ax.set_title(cls, fontsize=10)
        ax.set_xlabel("Posisi vertikal relatif\n(0 = atas kepala, 1 = kaki)",
                      fontsize=8)
    axes[0].set_ylabel("Jumlah")
    fig.suptitle("Validasi asumsi zona anatomis APD", y=1.03)
    return _save(fig, cfg.output_dir / "figures" / "zone_positions.png")


def render_annotated_samples(cfg: EDAConfig, images_df: pd.DataFrame,
                             boxes_df: pd.DataFrame) -> list:
    """
    Gambar ulang anotasi ground-truth pada sampel acak.

    Inspeksi visual manual tidak tergantikan oleh statistik: kesalahan
    anotasi sistematis sering hanya terlihat oleh mata.
    """
    random.seed(cfg.random_seed)
    with_boxes = images_df[images_df["n_boxes"] > 0]
    if with_boxes.empty:
        return []

    picks = with_boxes.sample(
        min(cfg.sample_images, len(with_boxes)), random_state=cfg.random_seed
    )
    saved = []

    for row in picks.itertuples():
        try:
            img = Image.open(row.image_path).convert("RGB")
        except Exception:  # noqa: BLE001
            continue

        boxes = boxes_df[
            (boxes_df["image_id"] == row.image_id) & (boxes_df["split"] == row.split)
        ]
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(img)
        ax.axis("off")

        for b in boxes.itertuples():
            color = PALETTE.get(b.class_name, "#a855f7")
            lw = 2.4 if b.class_name == PERSON_CLASS else 1.5
            ax.add_patch(patches.Rectangle(
                (b.x1 * img.width, b.y1 * img.height),
                b.w * img.width, b.h * img.height,
                fill=False, edgecolor=color, linewidth=lw,
            ))
            ax.text(b.x1 * img.width, b.y1 * img.height - 3, b.class_name,
                    fontsize=7, color="white",
                    bbox=dict(facecolor=color, edgecolor="none", pad=1))

        ax.set_title(f"{row.split}/{row.image_id}", fontsize=9)
        out = cfg.output_dir / "samples" / f"{row.split}_{row.image_id}.png"
        saved.append(_save(fig, out))

    return saved


def run(cfg: EDAConfig, images_df: pd.DataFrame, boxes_df: pd.DataFrame,
        dist: dict, geom: dict, assoc: dict) -> Dict[str, object]:
    figures: Dict[str, object] = {}
    """
    Titik masuk modul visualize: panggil semua fungsi plot_* di atas dan
    kumpulkan path hasilnya menjadi satu dict untuk disisipkan ke laporan.
    """

    counts = dist.get("_counts_df")
    if counts is not None and not counts.empty:
        figures["class_distribution"] = plot_class_distribution(counts, cfg)

    size_dist = geom.get("_size_dist_df")
    if size_dist is not None and not size_dist.empty:
        figures["size_distribution"] = plot_size_distribution(size_dist, cfg)

    if not boxes_df.empty:
        figures["area_boxplot"] = plot_area_boxplot(boxes_df, cfg)

    sens = assoc.get("sensitivity")
    if sens:
        figures["association_sensitivity"] = plot_association_sensitivity(
            pd.DataFrame(sens), cfg
        )

    assoc_df = assoc.get("_assoc_df")
    if assoc_df is not None and not assoc_df.empty:
        figures["zone_positions"] = plot_zone_positions(assoc_df, cfg)

    figures["samples"] = render_annotated_samples(cfg, images_df, boxes_df)
    return figures
