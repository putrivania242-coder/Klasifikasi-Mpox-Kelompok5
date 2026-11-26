
# Memeriksa ukuran gambar SEBELUM dan SESUDAH diproses getData.py (contoh: Train fold 1)
# Jalankan dari folder proyek: python cek_ukuran.py
import cv2 as cv
import matplotlib.pyplot as plt
from os import listdir, path
from config import FOLDER_ORI, JUMLAH_KELAS, UKURAN
from Utils.getData import getImageLabel, daftarKelas, EKSTENSI

folder = path.join(FOLDER_ORI.format(k=1), "Train")

# 1. Ukuran asli
ukuran_asli = set()
for kelas in daftarKelas(folder):
    for nama in listdir(path.join(folder, kelas)):
        if nama.lower().endswith(EKSTENSI):
            ukuran_asli.add(cv.imread(path.join(folder, kelas, nama)).shape)
print("SEBELUM:", len(ukuran_asli), "ukuran berbeda, contoh:", sorted(ukuran_asli)[:3])

# 2. Ukuran setelah getData.py: harus semuanya (100, 100, 3)
data = getImageLabel(folder, JUMLAH_KELAS, UKURAN)
print("SESUDAH:", set(tuple(gambar.shape) for gambar, label in data), "dari", len(data), "gambar")

# 3. Nilai piksel setelah dibagi 255: harus di antara 0 dan 1
gambar, label = data[0]
print("Nilai piksel terkecil =", round(gambar.min().item(), 3), "| terbesar =", round(gambar.max().item(), 3))

# 4. Lihat langsung: satu foto Monkeypox sebelum dan sesudah
fmpox = path.join(folder, "Monkeypox")
asli = cv.imread(path.join(fmpox, sorted(listdir(fmpox))[0]))
plt.figure(figsize=(8, 3.5))
plt.subplot(1, 2, 1); plt.imshow(cv.cvtColor(asli, cv.COLOR_BGR2RGB)); plt.title(f"Sebelum {asli.shape}")
plt.subplot(1, 2, 2); plt.imshow(cv.cvtColor(cv.resize(asli, (UKURAN, UKURAN)), cv.COLOR_BGR2RGB));
plt.title(f"Sesudah ({UKURAN}, {UKURAN}, 3)")
plt.tight_layout(); plt.savefig("cek_ukuran.png")
print("Gambar perbandingan disimpan: cek_ukuran.png")
