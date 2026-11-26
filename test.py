
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

from os import path
from torch.utils.data import DataLoader
from Utils.getData import getImageLabel
from Models.CNN import SimpleCNN
from config import *

from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score)


def uji_satu_fold(k):
    # Data Test hanya dipakai di sini, sekali, setelah pelatihan selesai
    data_uji = getImageLabel(path.join(FOLDER_ORI.format(k=k), "Test"), JUMLAH_KELAS, UKURAN)
    test_loader = DataLoader(data_uji, batch_size=BATCH_SIZE, shuffle=False)

    model = SimpleCNN(input_c=3, output=JUMLAH_KELAS)
    checkpoint = torch.load(f"SimpleCNN_fold{k}.pt", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    peluang, ground_truth = [], []
    with torch.no_grad():
        for src, trg in test_loader:
            src = torch.permute(src, (0, 3, 1, 2))
            peluang.extend(torch.softmax(model(src), dim=1).numpy())
            ground_truth.extend(torch.argmax(trg, dim=1).numpy())

    peluang = np.array(peluang)
    prediction = peluang.argmax(axis=1)

    # Confusion matrix
    kelas = data_uji.kelas
    cf_matrix = confusion_matrix(ground_truth, prediction, labels=range(JUMLAH_KELAS))
    plt.figure(figsize=(8, 6))
    sn.heatmap(pd.DataFrame(cf_matrix, index=kelas, columns=kelas),
               annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Prediksi")
    plt.ylabel("Label asli")
    plt.title(f"Fold {k} (epoch terbaik: {checkpoint['epoch']})")
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_fold{k}.png")
    plt.close()

    return {
        "fold": k,
        "accuracy": accuracy_score(ground_truth, prediction),
        "precision": precision_score(ground_truth, prediction, average='weighted', zero_division=0),
        "recall": recall_score(ground_truth, prediction, average='weighted', zero_division=0),
        "f1": f1_score(ground_truth, prediction, average='weighted', zero_division=0),
        "roc_auc": roc_auc_score(ground_truth, peluang, multi_class='ovr',
                                 average='weighted', labels=range(JUMLAH_KELAS)),
    }


def main():
    hasil = pd.DataFrame([uji_satu_fold(k) for k in FOLDS]).set_index("fold")
    hasil.loc["rata-rata"] = hasil.mean()
    hasil.loc["simpangan"] = hasil.iloc[:-1].std()
    print(hasil.round(3))
    hasil.round(4).to_csv("hasil_5fold.csv")


if __name__ == "__main__":
    main()
