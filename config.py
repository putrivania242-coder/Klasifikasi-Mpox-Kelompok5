FOLDER_ORI = "./Dataset/Original Images/Original Images/FOLDS/fold{k}"  # berisi Train/ Valid/ Test/
FOLDER_AUG = "./Dataset/Augmented Images/Augmented Images/FOLDS_AUG/fold{k}_AUG/Train"  # hanya Train
PAKAI_AUG = False        # True = data latih ditambah gambar augmentasi
 
FOLDS = [1, 2, 3, 4, 5] # untuk mencoba cepat, isi [1] saja
JUMLAH_KELAS = 6
UKURAN = 100            # semua gambar diubah menjadi 100 x 100 piksel
 
BATCH_SIZE = 32
EPOCH = 20
LEARNING_RATE = 0.0001  # besar langkah belajar model
