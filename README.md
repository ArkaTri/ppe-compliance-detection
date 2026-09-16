# Pemeriksa Kepatuhan APD Konstruksi

Sistem visi komputer yang menilai kelengkapan Alat Pelindung Diri **per pekerja**,
bukan sekadar menghitung objek dalam gambar.

**[Coba aplikasinya →](https://ppe-compliance-check.streamlit.app)**

---

## Masalah

Kepatuhan APD di lokasi konstruksi adalah kewajiban hukum di Indonesia — UU No. 1
Tahun 1970 tentang Keselamatan Kerja, PP No. 50 Tahun 2012 tentang SMK3, dan
Permenaker No. 8 Tahun 2010 tentang APD. Pemeriksaannya hari ini dilakukan manual:
safety officer berkeliling, mencatat, membuat laporan.

Pendekatan otomatis yang lazim adalah menghitung objek per kelas. Tetapi
penghitungan kelas **tidak dapat menentukan siapa yang melanggar.**

Contoh: sebuah gambar memuat 4 pekerja, terdeteksi 4 helm, 2 rompi, 2 tanpa-rompi.
Kesimpulan "2 orang tidak lengkap" hanya benar secara kebetulan. Bila satu pekerja
memegang helm cadangan, atau dua pekerja saling menutupi, logika itu runtuh — dan
runtuh secara diam-diam, karena keluarannya tetap terlihat masuk akal.

Pada dataset ini terdapat gambar nyata di mana jumlah box `helmet` melebihi jumlah
box `person`. Penghitungan kelas menghasilkan omong kosong di situ, tanpa gejala.

---

## Pendekatan: person-centric compliance

Sistem ini mengatribusikan setiap APD ke pekerja pemiliknya, lalu memberi vonis per
individu.

```
Tahap 1  deteksi person + APD pada gambar penuh
Tahap 2  potong tiap bbox pekerja, deteksi ulang pada potongan
Gabung   NMS per grup eksklusif (helmet/no-helmet saling meniadakan)
Asosiasi  containment ratio + bonus zona anatomis, penugasan greedy
Vonis    per pekerja: PATUH / MELANGGAR / PERLU TINJAU
```

### Tiga keputusan yang membentuk sistem

**Containment ratio, bukan IoU.** Box helm luasnya bisa hanya 5% dari box tubuh,
sehingga IoU maksimalnya juga sekitar 0,05 meski helm berada sepenuhnya di dalam
bbox pekerja. Memakai IoU standar membuat seluruh asosiasi gagal. Yang dipakai:
`area(irisan) / area(box APD)`.

**Fail-safe, bukan fail-silent.** Bila bagian tubuh terpotong tepi gambar atau
tertutup pekerja lain, sistem mengembalikan `PERLU TINJAU` — bukan `PATUH`. Sistem
keselamatan tidak boleh menyatakan aman sesuatu yang tidak diketahui.

**Menolak memberi angka yang menyesatkan.** Bila jumlah APD tanpa pemilik sebanding
dengan jumlah pekerja terdeteksi, tingkat kepatuhan tidak ditampilkan — angkanya
akan dihitung dari sampel yang tidak mewakili lokasi.

---

## Hasil

Divalidasi terhadap anotasi ground truth pada test set: 90 gambar, 333 keputusan
per bagian tubuh.

| Metrik | Nilai |
|---|---|
| **Presisi vonis "patuh"** | **0,972** |
| Presisi vonis "melanggar" | 0,859 |
| Recall pelanggaran | 0,873 |
| Beban tinjau manual | 4,8% |

Dari 64 pelanggaran ground truth: **86% tertangkap otomatis**, sisanya sebagian masuk
antrean tinjau manual, dan sekitar 6% lolos tanpa jejak.

Metrik yang paling penting adalah presisi vonis "patuh" — karena vonis itulah yang
paling berbahaya bila salah.

### Perjalanan metrik

| Tahap | Recall pelanggaran | Presisi |
|---|---|---|
| Baseline YOLOv12n @ 640 | 0,781 | 0,893 |
| + oversampling 10x | *ditolak* | — |
| + mosaic | *ditolak (melanggar panduan)* | — |
| + resolusi 960 | 0,781 | 0,893 |
| + augmentasi fotometrik | 0,857 | 0,871 |
| **+ kaskade dua tahap** | **0,873** | **0,859** |

---

## Temuan data yang menentukan arah

Pipeline EDA di `src/eda/` menjalankan tujuh pemeriksaan sebelum satu baris kode
training ditulis.

**Ketidakseimbangan 21,8x.** `no-helmet` hanya punya 129 instance dari total 7724
bounding box — dan merupakan kelas paling langka **sekaligus paling kecil** (median
luas 0,0062 vs `helmet` 0,0093). Dua kerugian menumpuk pada kelas yang paling kritis
secara keselamatan.

**Validation set hanya punya 11 instance `no-helmet`.** Satu deteksi meleset
menggeser recall sekitar 9 poin. Karena itu seluruh metrik per kelas dilaporkan
dengan interval kepercayaan bootstrap, bukan sebagai angka tunggal.

**Kelayakan asosiasi diuji lebih dulu.** Person coverage 98,6%, attach rate 95,8%,
ambiguity rate 0,6%. Ambiguity serendah itu berarti Hungarian assignment tidak
diperlukan — greedy berbasis skor sudah cukup. Menahan diri dari kompleksitas yang
tidak dibutuhkan data adalah keputusan, bukan kelalaian.

**Dua pasang gambar nyaris identik lintas split.** Pemeriksaan duplikat berbasis
hash biner melaporkan nol; audit perceptual hash (dHash) menemukan `ppe_0546`↔`ppe_0567`
dan `ppe_0606`↔`ppe_0591` dengan jarak Hamming 0. Dampaknya kecil pada 1206 gambar,
tetapi skor evaluasi sedikit optimistis dan itu dilaporkan apa adanya.

---

## Eksperimen yang ditolak

Tiga dari delapan hipotesis ditolak berdasarkan pengukuran. Semuanya terdokumentasi
di [notebook](capstone4_ppe_compliance.ipynb).

### Oversampling kelas minoritas 10x

Recall `no-helmet` justru **turun** 0,458 → 0,417, dan epoch terbaik bergeser dari
16 ke 11 — overfitting datang lebih cepat.

Penyebabnya: oversampling mengulang 49 gambar yang sama sepuluh kali. Tidak ada
informasi baru, hanya kesempatan lebih banyak untuk menghafalnya.

> *Class imbalance* dan *data scarcity* adalah dua masalah berbeda. Oversampling
> menyembuhkan yang pertama; masalah di sini yang kedua.

### Jalur inferensi dari ketiadaan APD

Rancangan awal: karena detector `no-helmet` lemah, sistem juga menyimpulkan
pelanggaran dari *ketiadaan* APD pada pekerja yang terdeteksi jelas.

Divalidasi terhadap ground truth, presisinya hanya **0,31** — dari 26 tuduhan, 18
pekerja yang sebenarnya patuh akan dituduh melanggar. Dipecah per bagian tubuh:
0,27 untuk kepala, 0,36 untuk torso.

Jalur ini diturunkan menjadi penanda tinjau manual, bukan vonis.

### Kalibrasi ulang ambang di bawah kaskade

Ambang per kelas awalnya disapu pada konfigurasi satu tahap, sementara produksi
menjalankan kaskade — distribusi keyakinannya berbeda. Kalibrasi ulang tampak logis.

Hasilnya recall turun 4,8 poin. Penyebabnya: sapuan F2 mengoptimalkan **deteksi
kotak**, sementara sistem dinilai pada **vonis per pekerja**. Kehilangan satu
`no-vest` di level kotak adalah satu false negative; di level vonis, seorang
pelanggar berubah menjadi patuh.

> Metrik proksi yang terlihat masuk akal bisa menyesatkan bila tidak sejajar dengan
> keputusan akhir yang sebenarnya diukur.

---

## Bug yang tidak menghasilkan gejala

Saat memvalidasi kaskade, presisi vonis pelanggaran anjlok dari 0,893 ke **0,643**
pada konfigurasi yang seharusnya identik dengan validasi sebelumnya.

Penyebabnya: pipeline mengonversi gambar ke array RGB (dibutuhkan untuk memotong
crop), lalu meneruskannya ke model. **Ultralytics memperlakukan array numpy sebagai
BGR.** Model menerima gambar dengan kanal merah dan biru tertukar.

Tidak ada error. Tidak ada peringatan. Program berjalan mulus dan angka keluar rapi.

Yang menyelamatkan: adanya baseline yang sudah diukur sebelumnya sebagai titik acuan.
Tanpa itu, penurunan ke 0,643 akan terbaca sebagai sifat data, bukan kerusakan.

> Bug yang tidak menghasilkan gejala hanya dapat ditemukan dengan membandingkan
> terhadap pengukuran sebelumnya. Disiplin mencatat baseline bukan formalitas.

---

## Keterbatasan yang diketahui

**Ketidaksesuaian domain.** Model dilatih pada rompi hi-vis potongan terbuka. Pada
foto pekerja migas berseragam coverall oranye reflektif, sistem menilai 5 dari 5
pekerja melanggar — padahal APD mereka benar, hanya jenisnya tidak pernah dilihat.

Modul `src/inference/domain_check.py` mengenali tanda tangan kegagalan ini — vonis
massal disertai keyakinan bukti rendah — dan memunculkan peringatan. Ia tidak
mengubah vonis per pekerja, hanya menambahkan konteks agar pengguna tidak menindak
laporan yang kemungkinan besar keliru.

**Kelangkaan `no-helmet`.** 129 instance di seluruh dataset. Augmentasi memperbanyak
variasi dari 49 gambar yang sama; kaskade memperbaiki cara melihat. Tidak satu pun
menambah pekerja tanpa helm yang benar-benar baru.

**Dataset eksternal dievaluasi, tidak dipakai.** Kandidat terbaik
(BAC_HIEN_CONSTRUCTION_SAFETY_2024, 19k gambar) taksonominya hanya memuat
`Helmet, No-Helmet, Person` — tanpa `vest`. Menambahkan gambar berisi pekerja
ber-rompi tanpa label rompi akan mengajari model menekan deteksi `vest`, kelas yang
justru sudah kuat. Perangkat penyaringnya tetap dibangun di
`src/training/external_data.py`, lengkap dengan penolakan taksonomi tidak lengkap
dan deteksi kebocoran nyaris-duplikat.

Sistem ini adalah **alat bantu penyaringan awal**, bukan pengganti inspeksi
keselamatan.

---

## Struktur repositori

```
src/
├── eda/           pipeline eksplorasi data — 7 modul, 1 pertanyaan per modul
├── training/      konfigurasi run, augmentasi offline, ablation, evaluasi
└── inference/     asosiasi, kepatuhan, kaskade, anotasi, audit, cek domain

app.py             aplikasi Streamlit
models/            bobot terpilih + ambang per kelas
run_eda.py         CLI eksplorasi data
run_training.py    CLI pelatihan + ablation
run_augment.py     CLI augmentasi fotometrik offline
run_external.py    CLI audit kebocoran + penggabungan dataset eksternal
```

Prinsip desain: satu modul menjawab satu pertanyaan; hanya `loader` yang menyentuh
disk sehingga modul analisis dapat diuji tanpa dataset asli; integritas diperiksa
sebelum statistik.

---

## Menjalankan

```bash
git clone https://github.com/ArkaTri/ppe-compliance-detection.git
cd ppe-compliance-detection
pip install -r requirements.txt
streamlit run app.py
```

Eksplorasi data dan pelatihan:

```bash
pip install -r requirements-eda.txt -r requirements-training.txt

python run_eda.py --dataset-root ./dataset --output-dir eda_output
python run_training.py --dataset-root ./dataset --runs all --epochs 100 --device 0
```

---

## Atribusi

Dataset: *construction safety* (Roboflow Universe, workspace `personal-project-kej16`),
bagian dari benchmark [RF100](https://rf100.org) yang disponsori Intel.
Lisensi **CC BY 4.0**.

Model: [Ultralytics YOLOv12](https://github.com/ultralytics/ultralytics), AGPL-3.0.
Penggunaan komersial memerlukan lisensi enterprise dari Ultralytics.

---

Capstone Project Modul 4 — Purwadhika AI Engineering
