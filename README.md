# EDA Pipeline — Construction Safety PPE Detection

Modul EDA untuk Capstone Project Module 4. Tujuannya bukan sekadar
menampilkan grafik, melainkan **menghasilkan keputusan teknis yang bisa
dipertanggungjawabkan** sebelum satu baris kode training ditulis.

## Struktur

```
ppe_eda/
├── run_eda.py              # orchestrator (CLI)
├── requirements.txt
└── src/eda/
    ├── config.py           # path, semantik kelas, ambang batas
    ├── loader.py           # folder YOLO  →  DataFrame
    ├── integrity.py        # kesehatan data (dijalankan pertama)
    ├── distribution.py     # sebaran kelas & ko-okurensi
    ├── geometry.py         # ukuran box  →  keputusan imgsz
    ├── association.py      # UJI KELAYAKAN person ↔ APD  ← modul penentu
    ├── visualize.py        # figur PNG untuk laporan & video
    └── report.py           # eda_report.md + eda_report.json
```

## Prinsip desain

1. **Satu modul, satu pertanyaan.** Tiap modul menjawab satu pertanyaan
   analitis dan tidak tahu-menahu urusan modul lain.
2. **Hanya `loader` yang menyentuh disk.** Modul lain bekerja di atas
   DataFrame, sehingga dapat diuji tanpa dataset asli.
3. **Integritas sebelum statistik.** Statistik di atas data kotor
   menghasilkan kesimpulan salah yang terlihat meyakinkan.
4. **Setiap keputusan punya angka pendukung.** Bagian *Keputusan Teknis*
   pada laporan menautkan tiap pilihan training ke temuan spesifik.

## Cara pakai

```bash
pip install -r requirements.txt

python run_eda.py \
  --dataset-root ./construction-safety \
  --output-dir eda_output
```

Di Google Colab:

```python
!pip install -q pandas numpy pillow matplotlib pyyaml tabulate
!python run_eda.py --dataset-root /content --output-dir /content/eda_output

from IPython.display import Markdown, display
display(Markdown(open('/content/eda_output/eda_report.md').read()))
```

Opsi lain: `--containment-threshold` (default 0.60), `--sample-images`,
`--skip-figures`, `--verbose`.

## Output

| Berkas | Isi |
|---|---|
| `eda_report.md` | Laporan lengkap, siap masuk README GitHub |
| `eda_report.json` | Versi mesin-terbaca untuk dikonsumsi modul training |
| `figures/*.png` | Grafik untuk laporan dan video penjelasan |
| `samples/*.png` | Anotasi ground-truth pada sampel acak |

## Metrik penentu

| Metrik | Arti | Ambang |
|---|---|---|
| `person_coverage` | Fraksi gambar ber-APD yang juga punya box `person` | ≥ 70% |
| `attach_rate` | Fraksi box APD yang menemukan person pemiliknya | ≥ 70% |
| `ambiguity_rate` | Fraksi box APD yang bisa diklaim >1 person | < 15% ideal |

Jika `person_coverage` atau `attach_rate` di bawah ambang, arsitektur
*person-centric compliance* harus dirancang ulang — dan laporan akan
menyebutkan alternatifnya.

## Catatan metodologis

Asosiasi memakai **containment ratio** (`area irisan / area box APD`),
bukan IoU. Box helm jauh lebih kecil daripada box tubuh, sehingga IoU
standar selalu bernilai kecil meski helm berada sepenuhnya di dalam
bbox person — dan akan menghasilkan kesimpulan yang salah.

## Atribusi

Dataset *construction safety* dari Roboflow Universe (workspace
`personal-project-kej16`), bagian dari benchmark RF100.
Lisensi **CC BY 4.0**.
