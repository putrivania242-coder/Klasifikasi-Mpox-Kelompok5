import cv2 as cv
import torch
import numpy as np
from os import listdir, path
from torch.utils.data import Dataset
 
EKSTENSI = ('.jpg', '.jpeg', '.png', '.bmp')
 
 
def daftarKelas(folder):
    # nama subfolder = nama kelas, diurutkan abjad (abaikan file tersembunyi seperti .DS_Store)
    return sorted(d for d in listdir(folder) if not d.startswith('.') and path.isdir(path.join(folder, d)))
 
 
class getImageLabel(Dataset):
    def __init__(self, folder, jumlah_kelas=6, ukuran=100):
        self.dataset = []
        self.kelas = daftarKelas(folder)
        to_one_hot = np.eye(jumlah_kelas)  # label one-hot, contoh kelas ke-1: [1, 0, 0, 0, 0, 0]
        for i, nama_kelas in enumerate(self.kelas):
            for nama_file in listdir(path.join(folder, nama_kelas)):
                if not nama_file.lower().endswith(EKSTENSI):
                    continue
                image = cv.imread(path.join(folder, nama_kelas, nama_file))
                image = cv.resize(image, (ukuran, ukuran))
                # disimpan sebagai uint8 (hemat memori); dibagi 255 saat diambil
                self.dataset.append([image, to_one_hot[i]])
 
    def __getitem__(self, item):
        feature, label = self.dataset[item]
        return torch.tensor(feature / 255, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
 
    def __len__(self):
        return len(self.dataset)