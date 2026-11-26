
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import platform
from datetime import datetime

from os import path
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import Adam
from Utils.getData import getImageLabel
from Models.CNN import SimpleCNN
from config import *


def catat(teks):
    # tampilkan di terminal DAN simpan ke train_log.txt
    print(teks, flush=True)
    with open("train_log.txt", "a") as f:
        f.write(teks + "\n")


def latih_satu_fold(k):
    # 1. Data latih dan validasi
    folder = FOLDER_ORI.format(k=k)
    data_latih = getImageLabel(
        path.join(folder, "Train"), JUMLAH_KELAS, UKURAN
    )

    if PAKAI_AUG:
        data_aug = getImageLabel(
            FOLDER_AUG.format(k=k), JUMLAH_KELAS, UKURAN
        )
        if data_aug.kelas != data_latih.kelas:
            raise ValueError(
                f"Nama kelas berbeda!\nOriginal : {data_latih.kelas}\n"
                f"Augmented: {data_aug.kelas}"
            )
        data_latih = ConcatDataset([data_latih, data_aug])

    data_valid = getImageLabel(
        path.join(folder, "Valid"), JUMLAH_KELAS, UKURAN
    )

    catat(
        f"=== FOLD {k}: {len(data_latih)} gambar latih, "
        f"{len(data_valid)} gambar validasi ==="
    )

    train_loader = DataLoader(
        data_latih, batch_size=BATCH_SIZE, shuffle=True
    )
    valid_loader = DataLoader(
        data_valid, batch_size=BATCH_SIZE, shuffle=False
    )

    # 2. Model baru untuk setiap fold
    torch.manual_seed(42)
    model = SimpleCNN(input_c=3, output=JUMLAH_KELAS)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    loss_function = nn.CrossEntropyLoss()

    loss_train_all, loss_valid_all = [], []
    loss_valid_terbaik = float("inf")

    for epoch in range(EPOCH):
        train_loss, valid_loss = 0, 0
        model.train()

        for src, trg in train_loader:
            src = torch.permute(src, (0, 3, 1, 2))
            pred = model(src)
            loss = loss_function(pred, trg)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            for src, trg in valid_loader:
                src = torch.permute(src, (0, 3, 1, 2))
                valid_loss += loss_function(
                    model(src), trg
                ).item()

        train_loss /= len(train_loader)
        valid_loss /= len(valid_loader)

        loss_train_all.append(train_loss)
        loss_valid_all.append(valid_loss)

        catat(
            f"fold {k} | epoch {epoch+1:>3} | "
            f"train loss = {train_loss:.4f} | "
            f"valid loss = {valid_loss:.4f}"
        )

        # 3. Simpan model dengan valid loss terkecil
        if valid_loss < loss_valid_terbaik:
            loss_valid_terbaik = valid_loss
            torch.save(
                {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict()
                },
                f"./SimpleCNN_fold{k}.pt"
            )

    plt.figure()
    plt.plot(
        range(1, EPOCH + 1),
        loss_train_all,
        color="#931a00",
        label='Training'
    )
    plt.plot(
        range(1, EPOCH + 1),
        loss_valid_all,
        color="#3399e6",
        label='Validation'
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Fold {k}")
    plt.legend()
    plt.savefig(f"./training_fold{k}.png")
    plt.close()


def main():
    open("train_log.txt", "w").close()
    mulai = datetime.now()

    catat(
        f"Mulai : {mulai:%Y-%m-%d %H:%M:%S} | "
        f"komputer: {platform.node()} | "
        f"PAKAI_AUG = {PAKAI_AUG}, EPOCH = {EPOCH}, FOLDS = {FOLDS}"
    )

    for k in FOLDS:
        latih_satu_fold(k)

    selesai = datetime.now()
    catat(
        f"Selesai : {selesai:%Y-%m-%d %H:%M:%S} | "
        f"lama pelatihan: {(selesai - mulai).seconds // 60} menit "
        f"{(selesai - mulai).seconds % 60} detik"
    )


if __name__ == "__main__":
    main()
