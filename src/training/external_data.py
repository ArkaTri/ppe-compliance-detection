"""
Penggabungan dataset eksternal - dengan pengaman terhadap dua kegagalan senyap.

Latar belakang:
    Kelas `no-helmet` hanya punya 129 instance di seluruh dataset, 94 di
    antaranya pada split train. Augmentasi memperbanyak variasi dari gambar
    yang sama; deteksi kaskade memperbaiki cara melihat. Tidak satu pun
    menambah pekerja tanpa helm yang benar-benar baru. Hanya data nyata
    yang dapat melakukannya.

Dua kegagalan yang dicegah modul ini:

1. ANOTASI PARSIAL - kegagalan paling berbahaya.
   Mayoritas dataset hardhat publik memakai taksonomi seperti `Hardhat`,
   `NO-Hardhat`, `Safety Vest`, tanpa kelas `person`. Bila gambar semacam
   itu dimasukkan ke train set, model melihat foto berisi manusia yang
   tidak punya label person sama sekali, lalu belajar MENEKAN deteksi
   person di wilayah seperti itu.

   Detector `person` adalah tulang punggung arsitektur person-centric -
   asosiasi APD dan seluruh jalur pelanggaran bergantung padanya. Merusak
   detector itu demi menambah beberapa instance `no-helmet` adalah
   pertukaran yang jelas merugikan.

   Karena itu: dataset eksternal ditolak bila taksonominya tidak mencakup
   `person`, berapa pun banyak kelas minoritas yang dikandungnya.

2. KEBOCORAN NYARIS-DUPLIKAT.
   Pemeriksaan integritas pada EDA hanya mendeteksi duplikat byte-identical.
   Dataset publik kerap berbagi sumber foto yang sama - gambar yang sama
   dengan kompresi, pemotongan, atau resolusi berbeda tidak akan terdeteksi
   sebagai duplikat biner, tetapi tetap membocorkan data uji ke data latih.
   Skor evaluasi lalu terlihat bagus secara palsu.

   Modul ini memakai perceptual hash (dHash) untuk menolak gambar eksternal
   yang mirip dengan isi split valid atau test.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Kelas yang WAJIB ada pada taksonomi sumber eksternal.
# Tanpa ini, gambar yang ditambahkan akan mengajari model untuk berhenti
# mendeteksi pekerja - lihat penjelasan di docstring modul.
REQUIRED_CLASSES: Tuple[str, ...] = ("person",)


# --------------------------------------------------------------------------
# Perceptual hash
# --------------------------------------------------------------------------

def dhash(path: Path, size: int = 8) -> Optional[int]:
    """
    Difference hash - tahan terhadap perubahan skala, kompresi, dan warna.

    Membandingkan kecerahan piksel bertetangga, bukan nilai absolutnya,
    sehingga gambar yang sama dengan kualitas JPEG berbeda menghasilkan
    hash yang nyaris identik.
    """
    try:
        with Image.open(path) as im:
            g = im.convert("L").resize((size + 1, size), Image.LANCZOS)
        a = np.asarray(g, dtype=np.int16)
        bits = (a[:, 1:] > a[:, :-1]).flatten()
        out = 0
        for b in bits:
            out = (out << 1) | int(b)
        return out
    except Exception as exc:  # noqa: BLE001
        logger.warning("Gagal hash %s: %s", path.name, exc)
        return None


def hamming(a: int, b: int) -> int:
    """Hitung jarak Hamming antar dua hash — makin kecil, makin mirip gambarnya."""
    return bin(a ^ b).count("1")


def _hash_dir(img_dir: Path) -> Dict[Path, int]:
    """Hitung perceptual hash untuk semua gambar dalam satu folder sekaligus."""
    out: Dict[Path, int] = {}
    if not img_dir.is_dir():
        return out
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() in IMAGE_EXTS:
            h = dhash(p)
            if h is not None:
                out[p] = h
    return out


# --------------------------------------------------------------------------
# Audit dataset yang sudah ada
# --------------------------------------------------------------------------

def find_near_duplicates(dataset_root: Path,
                         splits: Sequence[str] = ("train", "valid", "test"),
                         threshold: int = 6) -> Dict[str, object]:
    """
    Cari gambar yang nyaris identik LINTAS split pada dataset yang ada.

    Pemeriksaan integritas EDA hanya menangkap duplikat biner. Ini menangkap
    kasus yang jauh lebih umum: foto sumber yang sama dengan kompresi atau
    ukuran berbeda. Bila ditemukan antara train dan test, seluruh skor
    evaluasi menjadi optimistis palsu.
    """
    root = Path(dataset_root)
    hashes: Dict[str, Dict[Path, int]] = {
        s: _hash_dir(root / s / "images") for s in splits
    }

    pairs: List[dict] = []
    split_list = list(splits)
    for i, sa in enumerate(split_list):
        for sb in split_list[i + 1:]:
            for pa, ha in hashes[sa].items():
                for pb, hb in hashes[sb].items():
                    d = hamming(ha, hb)
                    if d <= threshold:
                        pairs.append({
                            "split_a": sa, "image_a": pa.name,
                            "split_b": sb, "image_b": pb.name,
                            "distance": d,
                        })

    return {
        "threshold": threshold,
        "n_images": {s: len(h) for s, h in hashes.items()},
        "n_cross_split_pairs": len(pairs),
        "pairs": pairs[:50],
        "verdict": ("BERSIH - tidak ada kebocoran nyaris-duplikat lintas split"
                    if not pairs else
                    f"PERINGATAN - {len(pairs)} pasang gambar nyaris identik "
                    "ditemukan lintas split. Skor evaluasi berpotensi "
                    "optimistis palsu."),
    }


# --------------------------------------------------------------------------
# Inspeksi sumber eksternal
# --------------------------------------------------------------------------

def inspect_external(external_root: Path) -> Dict[str, object]:
    """
    Periksa taksonomi dan sebaran kelas sumber eksternal SEBELUM digabung.

    Selalu jalankan ini lebih dulu. Keputusan pemetaan kelas harus dibuat
    dengan melihat nama kelas yang sebenarnya, bukan menebaknya.
    """
    import yaml
    root = Path(external_root)
    yaml_path = root / "data.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"data.yaml tidak ditemukan di {yaml_path}")

    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    names = data.get("names", [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]

    counts: Dict[str, int] = {n: 0 for n in names}
    n_images = 0
    for split in ("train", "valid", "test"):
        lbl_dir = root / split / "labels"
        if not lbl_dir.is_dir():
            continue
        for lp in lbl_dir.glob("*.txt"):
            n_images += 1
            for line in lp.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if not parts:
                    continue
                try:
                    cid = int(float(parts[0]))
                except ValueError:
                    continue
                if 0 <= cid < len(names):
                    counts[names[cid]] += 1

    return {
        "root": str(root),
        "class_names": names,
        "n_labeled_images": n_images,
        "instances_per_class": dict(sorted(counts.items(),
                                           key=lambda kv: -kv[1])),
        "hint": ("Bandingkan nama kelas di atas dengan skema proyek "
                 "['helmet','no-helmet','no-vest','person','vest'], lalu susun "
                 "class_map secara eksplisit. Perhatikan apakah tersedia kelas "
                 "setara `person` - tanpa itu, sumber ini TIDAK dapat dipakai."),
    }


# --------------------------------------------------------------------------
# Penggabungan
# --------------------------------------------------------------------------

def merge_external(external_root: Path, target_root: Path,
                   class_map: Dict[str, Optional[str]],
                   target_class_names: Sequence[str],
                   want_classes: Sequence[str] = ("no-helmet",),
                   protect_dataset: Optional[Path] = None,
                   max_images: Optional[int] = 400,
                   dup_threshold: int = 8,
                   seed: int = 42) -> Dict[str, object]:
    """
    Gabungkan gambar terpilih dari sumber eksternal ke split TRAIN saja.

    Argumen:
        class_map        pemetaan nama kelas eksternal -> nama kelas proyek.
                         Nilai None berarti kelas tersebut dibuang.
        want_classes     hanya gambar yang memuat salah satu kelas ini yang
                         diambil. Menambahkan seluruh dataset eksternal akan
                         menenggelamkan dataset asli dan menggeser domain.
        protect_dataset  akar dataset proyek; split valid dan test di sini
                         dipakai sebagai acuan penolakan nyaris-duplikat.

    Split valid dan test TIDAK PERNAH disentuh.
    """
    import random
    rng = random.Random(seed)

    ext_root = Path(external_root)
    tgt_root = Path(target_root)

    # -- 1. taksonomi -----------------------------------------------------
    mapped_targets = {v for v in class_map.values() if v}
    missing = [c for c in REQUIRED_CLASSES if c not in mapped_targets]
    if missing:
        return {
            "accepted": False,
            "reason": (
                f"Sumber eksternal tidak menyediakan kelas {missing}. "
                "Menambahkan gambar berisi pekerja tanpa label `person` akan "
                "mengajari model untuk menekan deteksi person - merusak "
                "fondasi arsitektur person-centric. Sumber ini tidak dapat "
                "dipakai; cari dataset yang taksonominya mencakup person."
            ),
        }

    import yaml
    ext_names = yaml.safe_load((ext_root / "data.yaml")
                               .read_text(encoding="utf-8")).get("names", [])
    if isinstance(ext_names, dict):
        ext_names = [ext_names[k] for k in sorted(ext_names)]

    unmapped = [n for n in ext_names if n not in class_map]
    if unmapped:
        return {
            "accepted": False,
            "reason": (f"Kelas eksternal berikut belum dipetakan: {unmapped}. "
                       "Petakan secara eksplisit (ke nama kelas proyek atau "
                       "None untuk dibuang) - jangan biarkan implisit."),
        }

    tgt_index = {n: i for i, n in enumerate(target_class_names)}
    # id eksternal -> id proyek (atau None bila dibuang)
    id_map: Dict[int, Optional[int]] = {}
    for i, n in enumerate(ext_names):
        t = class_map.get(n)
        id_map[i] = tgt_index[t] if t else None

    want_ids = {i for i, n in enumerate(ext_names)
                if class_map.get(n) in set(want_classes)}
    if not want_ids:
        return {"accepted": False,
                "reason": f"Tidak ada kelas eksternal yang memetakan ke {want_classes}."}

    # -- 2. acuan nyaris-duplikat ----------------------------------------
    protect_hashes: List[int] = []
    if protect_dataset:
        for split in ("valid", "test"):
            protect_hashes.extend(
                _hash_dir(Path(protect_dataset) / split / "images").values())
    logger.info("Acuan anti-kebocoran: %d gambar valid/test", len(protect_hashes))

    # -- 3. seleksi -------------------------------------------------------
    candidates: List[Tuple[Path, Path]] = []
    for split in ("train", "valid", "test"):
        img_dir, lbl_dir = ext_root / split / "images", ext_root / split / "labels"
        if not img_dir.is_dir():
            continue
        for ip in sorted(img_dir.iterdir()):
            if ip.suffix.lower() not in IMAGE_EXTS:
                continue
            lp = lbl_dir / f"{ip.stem}.txt"
            if not lp.exists():
                continue
            ids = set()
            for line in lp.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if parts:
                    try:
                        ids.add(int(float(parts[0])))
                    except ValueError:
                        pass
            if ids & want_ids:
                candidates.append((ip, lp))

    rng.shuffle(candidates)

    # -- 4. penyalinan dengan penolakan duplikat --------------------------
    dst_img = tgt_root / "train" / "images"
    dst_lbl = tgt_root / "train" / "labels"
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    n_added = n_rejected_dup = n_empty_after_map = 0
    added_instances: Dict[str, int] = {}

    for ip, lp in candidates:
        if max_images and n_added >= max_images:
            break

        if protect_hashes:
            h = dhash(ip)
            if h is not None and any(hamming(h, ph) <= dup_threshold
                                     for ph in protect_hashes):
                n_rejected_dup += 1
                continue

        # petakan ulang label
        lines: List[str] = []
        for line in lp.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cid = int(float(parts[0]))
            except ValueError:
                continue
            new_id = id_map.get(cid)
            if new_id is None:
                continue
            lines.append(" ".join([str(new_id)] + parts[1:5]))
            added_instances[target_class_names[new_id]] = \
                added_instances.get(target_class_names[new_id], 0) + 1

        if not lines:
            n_empty_after_map += 1
            continue

        stem = f"ext_{ip.stem}"
        shutil.copy2(ip, dst_img / f"{stem}{ip.suffix}")
        (dst_lbl / f"{stem}.txt").write_text("\n".join(lines) + "\n",
                                             encoding="utf-8")
        n_added += 1

    return {
        "accepted": True,
        "n_candidates": len(candidates),
        "n_added": n_added,
        "n_rejected_near_duplicate": n_rejected_dup,
        "n_empty_after_mapping": n_empty_after_map,
        "instances_added": dict(sorted(added_instances.items(),
                                       key=lambda kv: -kv[1])),
        "class_map": {k: v for k, v in class_map.items()},
        "note": ("Hanya split train yang ditambah. Split valid dan test tetap "
                 "utuh, sehingga perbandingan dengan model sebelumnya sahih."),
    }
