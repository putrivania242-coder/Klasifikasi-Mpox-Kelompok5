# Jalankan sekali setelah dataset diekstrak: python cek_dataset.py
from os import path
from config import FOLDER_ORI, FOLDER_AUG, FOLDS
from Utils.getData import daftarKelas, EKSTENSI
from os import listdir
 
 
def hitung(folder):
    if not path.isdir(folder):
        print("   TIDAK DITEMUKAN:", folder)
        return
    kelas = daftarKelas(folder)
    isi = {k: sum(f.lower().endswith(EKSTENSI) for f in listdir(path.join(folder, k))) for k in kelas}
    print("   {:<72} {} kelas, total {}".format(folder, len(kelas), sum(isi.values())))
    print("      ", isi)
 
 
for k in FOLDS:
    print("FOLD", k)
    for bagian in ["Train", "Valid", "Test"]:
        hitung(path.join(FOLDER_ORI.format(k=k), bagian))
    hitung(FOLDER_AUG.format(k=k))
