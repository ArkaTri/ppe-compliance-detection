# Bobot model

Folder ini memuat bobot model terpilih dan konfigurasi ambangnya.

| Berkas | Asal |
|---|---|
| `best.pt` | `training_output/imgsz_960/weights/best.pt` |
| `evaluation_test.json` | `training_output/imgsz_960/evaluation_test.json` |

`evaluation_test.json` bukan pelengkap. Aplikasi membaca ambang per kelas
dari berkas ini. Tanpa berkas tersebut, aplikasi jatuh ke nilai bawaan di
`src/inference/config.py`, dan model akan berjalan dengan ambang yang
berbeda dari yang dioptimalkan saat evaluasi.

Salin keduanya ke sini sebelum menjalankan `app.py`.
