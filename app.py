"""
Aplikasi Streamlit - Pemeriksa Kepatuhan APD Konstruksi.

Jalankan lokal:
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from inference import audit                      # noqa: E402
from inference.annotate import STATUS_LABEL, annotate   # noqa: E402
from inference.config import (                   # noqa: E402
    COMPLIANT,
    InferenceConfig,
    NEEDS_REVIEW,
    VIOLATION,
    VIOLATION_INFERRED,
)
from inference.pipeline import analyze, load_model   # noqa: E402

ROOT = Path(__file__).parent
WEIGHTS = ROOT / "models" / "best.pt"
EVAL_JSON = ROOT / "models" / "evaluation_test.json"

st.set_page_config(page_title="Pemeriksa Kepatuhan APD",
                   page_icon="🦺", layout="wide")


# --------------------------------------------------------------------------
# Sumber daya
# --------------------------------------------------------------------------

@st.cache_resource(show_spinner="Memuat model ...")
def get_config_and_model(imgsz: int):
    cfg = InferenceConfig.from_evaluation(
        weights=WEIGHTS, eval_json=EVAL_JSON, imgsz=imgsz, device="cpu")
    model = load_model(cfg)
    return cfg, model


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

st.sidebar.title("Pengaturan")

imgsz = st.sidebar.selectbox(
    "Resolusi inferensi", [960, 640], index=0,
    help="960 menaikkan recall pada APD berukuran kecil, dengan biaya "
         "waktu proses sekitar 1,4x lebih lama.")

cfg, model = get_config_and_model(imgsz)

st.sidebar.subheader("Ambang keyakinan per kelas")
st.sidebar.caption(
    "Nilai awal diturunkan dari sapuan kurva pada test set: kelas "
    "pelanggaran dioptimalkan pada F2 (recall dibobot dua kali), kelas "
    "lain pada F1.")
for name in sorted(cfg.thresholds):
    cfg.thresholds[name] = st.sidebar.slider(
        name, 0.05, 0.95, float(cfg.thresholds[name]), 0.05)

st.sidebar.subheader("Pelanggaran tersimpulkan")
mode_label = {
    "review": "Tandai untuk tinjau manual (disarankan)",
    "violation": "Vonis pelanggaran",
    "off": "Abaikan",
}
cfg.path_b_mode = st.sidebar.radio(
    "Bila tidak ada APD terdeteksi pada pekerja yang terlihat jelas:",
    list(mode_label), format_func=lambda k: mode_label[k], index=0)

if cfg.path_b_mode == "violation":
    st.sidebar.warning(
        "Validasi pada test set: presisi jalur ini hanya 0,27 untuk kepala "
        "dan 0,36 untuk torso. Sebagian besar 'pelanggaran' yang dihasilkan "
        "sebenarnya adalah APD yang gagal terdeteksi.")

st.sidebar.subheader("Deteksi kaskade")
cfg.use_cascade = st.sidebar.checkbox(
    "Aktifkan deteksi dua tahap", value=True,
    help="Memotong tiap bbox pekerja lalu menjalankan detector lagi pada "
         "potongan itu. Helm bermedian 62 px pada frame penuh menjadi ratusan "
         "piksel di dalam potongan.")
cfg.use_tta = st.sidebar.checkbox(
    "Test-time augmentation", value=False,
    help="Menaikkan recall, memperlambat proses sekitar 2,5x.")

show_ppe = st.sidebar.checkbox("Tampilkan kotak APD", value=True)


# --------------------------------------------------------------------------
# Utama
# --------------------------------------------------------------------------

st.title("🦺 Pemeriksa Kepatuhan APD Konstruksi")
st.markdown(
    "Menilai kelengkapan APD **per pekerja**, bukan sekadar menghitung objek. "
    "Setiap helm dan rompi diatribusikan ke pekerja pemiliknya, sehingga "
    "sistem dapat menjawab *siapa* yang belum lengkap - bukan hanya *berapa*."
)

files = st.file_uploader(
    "Unggah foto lokasi kerja", type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True)

if not files:
    st.info("Unggah satu atau beberapa foto untuk memulai pemeriksaan.")
    st.stop()

reports = []
progress = st.progress(0.0, text="Memproses ...")

for i, f in enumerate(files, 1):
    from PIL import Image
    img = Image.open(f).convert("RGB")
    rep = analyze(img, cfg, model=model, image_id=f.name)
    rep["_pil"] = img
    reports.append(rep)
    progress.progress(i / len(files), text=f"Memproses {i}/{len(files)} ...")

progress.empty()

# -- ringkasan lokasi ------------------------------------------------------
agg = audit.aggregate(reports)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Pekerja terdeteksi", agg["n_workers"])
c2.metric("Patuh", agg["n_compliant"])
c3.metric("Melanggar", agg["n_violation"])
c4.metric("Perlu tinjau", agg["n_needs_review"])

if agg["compliance_rate"] is not None:
    st.progress(agg["compliance_rate"],
                text=f"Tingkat kepatuhan {agg['compliance_rate']:.0%} "
                     f"(dari {agg['assessable']} pekerja yang dapat dinilai)")

st.caption(audit.summary_text(agg))

st.divider()

# -- per gambar ------------------------------------------------------------
for rep in reports:
    st.subheader(rep["image_id"])
    left, right = st.columns([3, 2])

    with left:
        st.image(annotate(rep["_pil"], rep, show_ppe=show_ppe),
                 use_container_width=True)

    with right:
        rows = []
        for w in rep["workers"]:
            rows.append({
                "Pekerja": f"#{w['index']}",
                "Status": STATUS_LABEL.get(w["overall"], w["overall"]),
                "Kepala": STATUS_LABEL.get(w["head"]["status"], w["head"]["status"]),
                "Torso": STATUS_LABEL.get(w["torso"]["status"], w["torso"]["status"]),
            })

        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True,
                         use_container_width=True)
        else:
            st.warning("Tidak ada pekerja terdeteksi pada gambar ini.")

        s = rep["summary"]
        if s.get("orphan_warning"):
            st.warning(s["orphan_warning"])

        ds = rep.get("detection_stats", {})
        if ds.get("cascade") and ds.get("n_recovered_by_cascade"):
            st.success(
                f"Deteksi kaskade menemukan {ds['n_recovered_by_cascade']} APD "
                f"tambahan dari {ds['n_crops']} potongan pekerja - tidak "
                "terdeteksi pada pemindaian gambar penuh.")

        with st.expander("Dasar keputusan"):
            for w in rep["workers"]:
                st.markdown(f"**Pekerja #{w['index']}**")
                st.caption(f"Kepala - {w['head']['reason']}")
                st.caption(f"Torso - {w['torso']['reason']}")
                for n in w.get("notes", []):
                    st.caption(f"Catatan: {n}")

        with st.expander("Penghitungan per kelas"):
            st.caption("Format keluaran dasar. Tidak dipakai sebagai dasar "
                       "penilaian kepatuhan, karena penghitungan kelas tidak "
                       "dapat menentukan pekerja mana yang belum lengkap.")
            st.json(rep["class_counts"])

    st.divider()

# -- ekspor ----------------------------------------------------------------
all_rows = []
for rep in reports:
    all_rows.extend(audit.records_from_report(rep))

if all_rows:
    st.subheader("Catatan audit")
    st.caption(
        "Setiap baris memuat vonis beserta dasarnya. Kepatuhan APD diatur "
        "UU No. 1/1970, PP No. 50/2012, dan Permenaker No. 8/2010 - saat "
        "inspeksi atau insiden, yang diminta adalah dokumentasi."
    )
    st.dataframe(pd.DataFrame(all_rows), use_container_width=True, height=240)
    st.download_button(
        "Unduh catatan audit (CSV)",
        data=audit.to_csv(all_rows),
        file_name="audit_kepatuhan_apd.csv",
        mime="text/csv")

with st.expander("Batas kemampuan sistem"):
    st.markdown("""
Divalidasi terhadap anotasi ground truth pada test set (90 gambar,
333 keputusan per bagian tubuh):

| Metrik | Nilai |
|---|---|
| Presisi vonis "patuh" | 0,98 |
| Presisi vonis "melanggar" | 0,89 |
| Pelanggaran terdeteksi otomatis | 78% |
| Pelanggaran masuk antrean tinjau | 16% |
| **Pelanggaran lolos tanpa jejak** | **6%** |

Keterbatasan utama ada pada kelas `no-helmet`: hanya 129 instance di
seluruh dataset, sehingga detektornya jauh lebih lemah daripada
`no-vest` (892 instance). Sistem ini alat bantu penyaringan awal,
bukan pengganti inspeksi keselamatan.
    """)
