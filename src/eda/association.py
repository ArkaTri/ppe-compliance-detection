"""
Association: modul PENENTU KEPUTUSAN dari seluruh EDA ini.

Seluruh desain aplikasi bertumpu pada satu asumsi: setiap box APD dapat
dipetakan ke seorang `person` tertentu. Jika asumsi itu tidak didukung data,
arsitektur person-centric harus dirancang ulang SEBELUM training dimulai,
bukan setelahnya.

Modul ini menguji asumsi tersebut pada anotasi ground-truth. Yang diukur:

  1. person_coverage  - berapa fraksi gambar ber-APD yang juga punya box person
  2. attach_rate      - berapa fraksi box APD yang menemukan person pemiliknya
  3. ambiguity_rate   - berapa fraksi box APD yang bisa diklaim >1 person
  4. zona vertikal    - apakah helmet benar berada di kepala, vest di torso

Catatan metodologis penting: containment ratio dipakai, BUKAN IoU.
    containment = area(irisan) / area(box APD)
Box helm jauh lebih kecil dari box tubuh, sehingga IoU standar akan selalu
bernilai kecil dan menyesatkan meski helm berada sepenuhnya di dalam person.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from .config import (
    EDAConfig,
    EXPECTED_ZONES,
    HEAD_CLASSES,
    PERSON_CLASS,
    PPE_CLASSES,
    TORSO_CLASSES,
)


def containment(ppe: pd.Series, person: pd.Series) -> float:
    """Fraksi luas box APD yang berada di dalam box person."""
    ix1 = max(ppe["x1"], person["x1"])
    iy1 = max(ppe["y1"], person["y1"])
    ix2 = min(ppe["x2"], person["x2"])
    iy2 = min(ppe["y2"], person["y2"])

    iw, ih = ix2 - ix1, iy2 - iy1
    if iw <= 0 or ih <= 0:
        return 0.0

    ppe_area = (ppe["x2"] - ppe["x1"]) * (ppe["y2"] - ppe["y1"])
    if ppe_area <= 0:
        return 0.0
    return float((iw * ih) / ppe_area)


def relative_vertical_position(ppe: pd.Series, person: pd.Series) -> float:
    """
    Posisi pusat APD terhadap tinggi person: 0.0 = ubun-ubun, 1.0 = kaki.

    Distribusi nilai ini yang memvalidasi (atau membantah) hipotesis zona
    di config.EXPECTED_ZONES.
    """
    ph = person["y2"] - person["y1"]
    if ph <= 0:
        return float("nan")
    return float((ppe["yc"] - person["y1"]) / ph)


def analyze_image(group: pd.DataFrame, threshold: float) -> List[dict]:
    """Hitung kandidat asosiasi untuk satu gambar."""
    persons = group[group["class_name"] == PERSON_CLASS]
    ppes = group[group["class_name"].isin(PPE_CLASSES)]

    results: List[dict] = []
    for _, ppe in ppes.iterrows():
        scores = []
        for pidx, person in persons.iterrows():
            c = containment(ppe, person)
            if c >= threshold:
                scores.append((pidx, c, relative_vertical_position(ppe, person)))

        scores.sort(key=lambda t: t[1], reverse=True)
        results.append(
            {
                "split": ppe["split"],
                "image_id": ppe["image_id"],
                "ppe_class": ppe["class_name"],
                "n_persons_in_image": int(len(persons)),
                "n_candidate_owners": len(scores),
                "matched": len(scores) > 0,
                "ambiguous": len(scores) > 1,
                "best_containment": scores[0][1] if scores else 0.0,
                "rel_y": scores[0][2] if scores else float("nan"),
            }
        )
    return results


def zone_validation(assoc_df: pd.DataFrame) -> Dict[str, dict]:
    """
    Uji apakah asumsi zona anatomis benar-benar berlaku pada data.

    Jika `within_expected_zone` rendah, jangan pakai constraint geometris
    sebagai filter keras - ia akan membuang asosiasi yang valid.
    """
    out: Dict[str, dict] = {}
    matched = assoc_df[assoc_df["matched"]]

    for cls in PPE_CLASSES:
        sub = matched[matched["ppe_class"] == cls]["rel_y"].dropna()
        if sub.empty:
            continue
        lo, hi = EXPECTED_ZONES["head" if cls in HEAD_CLASSES else "torso"]
        out[cls] = {
            "n": int(len(sub)),
            "rel_y_p05": round(float(np.percentile(sub, 5)), 3),
            "rel_y_p50": round(float(sub.median()), 3),
            "rel_y_p95": round(float(np.percentile(sub, 95)), 3),
            "expected_zone": [lo, hi],
            "within_expected_zone": round(float(((sub >= lo) & (sub <= hi)).mean()), 4),
        }
    return out


def sensitivity_sweep(boxes_df: pd.DataFrame,
                      thresholds=(0.3, 0.5, 0.6, 0.7, 0.8, 0.9)) -> pd.DataFrame:
    """
    Attach rate dan ambiguity rate pada berbagai nilai ambang containment.

    Ini yang menghasilkan justifikasi angka: ambang dipilih dari kurva,
    bukan ditebak. Ambang ideal = attach rate masih tinggi sementara
    ambiguity rate sudah turun.
    """
    rows = []
    for th in thresholds:
        recs = []
        for _, g in boxes_df.groupby(["split", "image_id"], sort=False):
            recs.extend(analyze_image(g, th))
        if not recs:
            continue
        df = pd.DataFrame(recs)
        rows.append({
            "threshold": th,
            "attach_rate": round(float(df["matched"].mean()), 4),
            "ambiguity_rate": round(float(df["ambiguous"].mean()), 4),
        })
    return pd.DataFrame(rows)


def run(cfg: EDAConfig, boxes_df: pd.DataFrame) -> dict:
    if boxes_df.empty or PERSON_CLASS not in set(boxes_df["class_name"]):
        return {
            "feasible": False,
            "reason": f"Kelas '{PERSON_CLASS}' tidak ditemukan pada anotasi. "
                      "Arsitektur person-centric tidak dapat dibangun dari "
                      "ground-truth ini.",
        }

    # -- 1. cakupan person pada gambar yang memuat APD ---------------------
    per_image = boxes_df.groupby(["split", "image_id"])["class_name"].apply(set)
    has_ppe = per_image.apply(lambda s: bool(s & set(PPE_CLASSES)))
    has_person = per_image.apply(lambda s: PERSON_CLASS in s)

    ppe_images = per_image[has_ppe]
    person_coverage = float(has_person[has_ppe].mean()) if len(ppe_images) else 0.0

    # -- 2. asosiasi pada ambang default -----------------------------------
    records: List[dict] = []
    for _, g in boxes_df.groupby(["split", "image_id"], sort=False):
        records.extend(analyze_image(g, cfg.containment_threshold))
    assoc_df = pd.DataFrame(records)

    if assoc_df.empty:
        return {"feasible": False, "reason": "Tidak ada box APD pada dataset."}

    attach_rate = float(assoc_df["matched"].mean())
    ambiguity_rate = float(assoc_df["ambiguous"].mean())

    per_class_attach = (
        assoc_df.groupby("ppe_class")["matched"].mean().round(4).to_dict()
    )
    orphan_by_class = (
        assoc_df[~assoc_df["matched"]].groupby("ppe_class").size().to_dict()
    )

    # -- 3. verdict --------------------------------------------------------
    feasible = (
        person_coverage >= cfg.min_person_coverage
        and attach_rate >= cfg.min_ppe_attach_rate
    )

    if feasible:
        verdict = ("GO - asosiasi person-centric didukung ground-truth. "
                   "Lanjutkan desain association engine.")
    elif person_coverage < cfg.min_person_coverage:
        verdict = ("NO-GO - box 'person' tidak dilabeli secara konsisten pada "
                   "gambar ber-APD. Pertimbangkan pendekatan alternatif: "
                   "turunkan person dari box helmet/vest, atau pakai detector "
                   "person pretrained COCO sebagai lapisan terpisah.")
    else:
        verdict = ("BERSYARAT - person tersedia, tetapi banyak box APD tidak "
                   "menemukan pemilik pada ambang saat ini. Periksa "
                   "sensitivity_sweep dan pertimbangkan menurunkan ambang.")

    return {
        "feasible": feasible,
        "verdict": verdict,
        "containment_threshold": cfg.containment_threshold,
        "n_images_with_ppe": int(len(ppe_images)),
        "person_coverage": round(person_coverage, 4),
        "attach_rate": round(attach_rate, 4),
        "ambiguity_rate": round(ambiguity_rate, 4),
        "attach_rate_per_class": per_class_attach,
        "orphan_ppe_by_class": {k: int(v) for k, v in orphan_by_class.items()},
        "zone_validation": zone_validation(assoc_df),
        "sensitivity": sensitivity_sweep(boxes_df).to_dict("records"),
        "_assoc_df": assoc_df,
    }
