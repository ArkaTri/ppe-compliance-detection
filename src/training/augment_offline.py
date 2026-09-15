"""
Augmentasi fotometrik offline - memperbanyak data TANPA mengubah geometri.

Latar belakang masalah:
    Seluruh eksperimen menunjukkan model overfit cepat - epoch terbaik pada
    13-24 dari 100 epoch, sementara train loss terus turun dan val mAP diam.
    Run `mosaic_ablation` memperbaikinya (epoch terbaik bergeser ke 24),
    tetapi mosaic adalah augmentasi geometris yang dilarang panduan capstone.

Pendekatan:
    Panduan justru MENYARANKAN augmentasi non-geometris: noise, pencahayaan,
    kontras. Teknik itu belum dimanfaatkan - konfigurasi saat ini hanya
    memakai hsv bawaan, dan albumentations otomatis Ultralytics berjalan pada
    p=0.01 sehingga praktis tidak aktif.

    Modul ini membangkitkan varian fotometrik dan menuliskannya ke disk
    sebagai gambar train tambahan. Karena tidak ada transformasi geometris,
    file label disalin apa adanya - koordinat bbox tetap sahih.

Keuntungan pendekatan offline dibanding menyetel parameter Ultralytics:
    1. Hasilnya dapat dilihat dan diperiksa sebelum training.
    2. Tidak perlu melawan internal Ultralytics.
    3. Varian untuk kelas minoritas dapat diperbanyak lebih agresif -
       oversampling dan penambahan variasi sekaligus, bukan sekadar
       mengulang gambar yang sama seperti eksperimen sebelumnya.

Transformasi dipilih agar meniru kondisi nyata lokasi konstruksi:
    pencahayaan buruk di terowongan, kamera CCTV bernoise, lensa berdebu,
    kontras ekstrem antara area terbuka dan bayangan.
"""

from __future__ import annotations

import logging
import random
import shutil
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# --------------------------------------------------------------------------
# Transformasi fotometrik (tidak ada yang mengubah geometri)
# --------------------------------------------------------------------------

def brightness(img: Image.Image, rng: random.Random) -> Image.Image:
    """Meniru variasi pencahayaan: terowongan gelap sampai area terik."""
    return ImageEnhance.Brightness(img).enhance(rng.uniform(0.45, 1.55))


def contrast(img: Image.Image, rng: random.Random) -> Image.Image:
    return ImageEnhance.Contrast(img).enhance(rng.uniform(0.55, 1.65))


def saturation(img: Image.Image, rng: random.Random) -> Image.Image:
    """Rompi hi-vis tampil berbeda di bawah lampu natrium vs sinar matahari."""
    return ImageEnhance.Color(img).enhance(rng.uniform(0.35, 1.60))


def gaussian_noise(img: Image.Image, rng: random.Random) -> Image.Image:
    """Meniru sensor CCTV pada cahaya rendah."""
    arr = np.asarray(img).astype(np.float32)
    sigma = rng.uniform(4, 22)
    arr += np.random.default_rng(rng.randint(0, 2**31)).normal(0, sigma, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def blur(img: Image.Image, rng: random.Random) -> Image.Image:
    """Lensa berdebu, fokus meleset, atau gerakan halus."""
    return img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.6, 2.0)))


def jpeg_artifact(img: Image.Image, rng: random.Random) -> Image.Image:
    """Kompresi berat - lazim pada rekaman CCTV yang dikirim lewat jaringan."""
    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=rng.randint(18, 55))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def color_temperature(img: Image.Image, rng: random.Random) -> Image.Image:
    """Pergeseran suhu warna: lampu pijar hangat vs LED dingin."""
    arr = np.asarray(img).astype(np.float32)
    shift = rng.uniform(-28, 28)
    arr[..., 0] = np.clip(arr[..., 0] + shift, 0, 255)        # merah
    arr[..., 2] = np.clip(arr[..., 2] - shift, 0, 255)        # biru
    return Image.fromarray(arr.astype(np.uint8))


def gamma(img: Image.Image, rng: random.Random) -> Image.Image:
    """Kompresi detail di area gelap atau terang - bukan sekadar brightness."""
    g = rng.uniform(0.55, 1.75)
    lut = [min(255, int((i / 255.0) ** g * 255)) for i in range(256)]
    return img.point(lut * 3)


TRANSFORMS: Dict[str, Callable] = {
    "brightness": brightness,
    "contrast": contrast,
    "saturation": saturation,
    "noise": gaussian_noise,
    "blur": blur,
    "jpeg": jpeg_artifact,
    "color_temp": color_temperature,
    "gamma": gamma,
}


def apply_chain(img: Image.Image, rng: random.Random,
                n_ops: Tuple[int, int] = (2, 3)) -> Tuple[Image.Image, List[str]]:
    """
    Terapkan beberapa transformasi berurutan.

    Menggabungkan 2-3 transformasi menghasilkan variasi yang jauh lebih kaya
    daripada satu transformasi tunggal, tanpa membuat gambar menjadi tidak
    realistis.
    """
    k = rng.randint(*n_ops)
    names = rng.sample(list(TRANSFORMS), k)
    out = img
    for name in names:
        out = TRANSFORMS[name](out, rng)
    return out, names


# --------------------------------------------------------------------------
# Pembangkitan dataset
# --------------------------------------------------------------------------

def _classes_in_label(label_path: Path) -> Set[int]:
    if not label_path.exists():
        return set()
    ids: Set[int] = set()
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if parts:
            try:
                ids.add(int(float(parts[0])))
            except ValueError:
                continue
    return ids


def generate(dataset_root: Path, output_root: Path,
             minority_class_ids: Optional[Set[int]] = None,
             n_variants: int = 1, n_variants_minority: int = 4,
             seed: int = 42) -> Dict[str, object]:
    """
    Bangun train set yang diperluas di `output_root`.

    Gambar asli disalin, lalu ditambahi varian fotometrik. Gambar yang memuat
    kelas minoritas mendapat lebih banyak varian - inilah bedanya dengan
    oversampling sederhana: yang ditambahkan adalah variasi baru, bukan
    duplikat gambar yang sama.

    Split valid dan test TIDAK disentuh. Mengaugmentasi data evaluasi akan
    membuat metrik tidak lagi mencerminkan kondisi nyata.
    """
    dataset_root = Path(dataset_root)
    output_root = Path(output_root)
    minority_class_ids = minority_class_ids or set()

    src_img = dataset_root / "train" / "images"
    src_lbl = dataset_root / "train" / "labels"
    dst_img = output_root / "train" / "images"
    dst_lbl = output_root / "train" / "labels"
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    images = sorted(p for p in src_img.iterdir()
                    if p.suffix.lower() in IMAGE_EXTS)

    n_original = n_generated = n_minority_images = 0
    op_counter: Dict[str, int] = {}

    for img_path in images:
        lbl_path = src_lbl / f"{img_path.stem}.txt"

        # salin yang asli
        shutil.copy2(img_path, dst_img / img_path.name)
        if lbl_path.exists():
            shutil.copy2(lbl_path, dst_lbl / lbl_path.name)
        n_original += 1

        classes = _classes_in_label(lbl_path)
        is_minority = bool(classes & minority_class_ids)
        if is_minority:
            n_minority_images += 1

        k = n_variants_minority if is_minority else n_variants
        if k <= 0:
            continue

        try:
            base = Image.open(img_path).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Gagal membuka %s: %s", img_path.name, exc)
            continue

        for i in range(k):
            aug, ops = apply_chain(base, rng)
            stem = f"{img_path.stem}_aug{i}"
            aug.save(dst_img / f"{stem}.jpg", quality=92)
            # label disalin apa adanya - tidak ada transformasi geometris,
            # sehingga seluruh koordinat bbox tetap sahih
            if lbl_path.exists():
                shutil.copy2(lbl_path, dst_lbl / f"{stem}.txt")
            n_generated += 1
            for op in ops:
                op_counter[op] = op_counter.get(op, 0) + 1

    # tautkan valid dan test apa adanya (symlink bila bisa, salin bila tidak)
    for split in ("valid", "test"):
        src = dataset_root / split
        dst = output_root / split
        if src.is_dir() and not dst.exists():
            try:
                dst.symlink_to(src.resolve(), target_is_directory=True)
            except OSError:
                shutil.copytree(src, dst)

    # data.yaml turunan
    src_yaml = dataset_root / "data.yaml"
    if src_yaml.exists():
        import yaml
        data = yaml.safe_load(src_yaml.read_text(encoding="utf-8"))
        data["train"] = str(dst_img.resolve())
        data["val"] = str((output_root / "valid" / "images").resolve())
        data["test"] = str((output_root / "test" / "images").resolve())
        (output_root / "data.yaml").write_text(
            yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    summary = {
        "output_root": str(output_root),
        "data_yaml": str(output_root / "data.yaml"),
        "n_original": n_original,
        "n_generated": n_generated,
        "n_total": n_original + n_generated,
        "expansion": round((n_original + n_generated) / max(n_original, 1), 2),
        "n_minority_images": n_minority_images,
        "variants_normal": n_variants,
        "variants_minority": n_variants_minority,
        "transform_usage": dict(sorted(op_counter.items(),
                                       key=lambda kv: -kv[1])),
    }
    logger.info("Augmentasi selesai: %d asli + %d varian = %d (%.2fx)",
                n_original, n_generated, summary["n_total"],
                summary["expansion"])
    return summary
