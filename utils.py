import numpy as np
from glob import iglob
from PIL import Image
import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt


def load_higgs_dataset():
    dataset = pd.read_csv("HIGGS_reduced.csv", header=None)

    Y = dataset.iloc[:, 1:2].to_numpy().astype(np.float32)
    X = dataset.iloc[:, 2:].to_numpy().astype(np.float32)

    return X, Y


def create_plots():
    bn_acc_hist = np.load("arrays/bn_acc_hist.npy")
    bn_val_acc_hist = np.load("arrays/bn_val_acc_hist.npy")
    no_acc_hist = np.load("arrays/no_acc_hist.npy")
    no_val_acc_hist = np.load("arrays/no_val_acc_hist.npy")

    bn_loss_hist = np.load("arrays/bn_loss_hist.npy")
    bn_val_loss_hist = np.load("arrays/bn_val_loss_hist.npy")
    no_loss_hist = np.load("arrays/no_loss_hist.npy")
    no_val_loss_hist = np.load("arrays/no_val_loss_hist.npy")

    x = np.arange(1, bn_acc_hist.shape[0] + 1)
    
    plt.figure().clear()
    plt.title("BN vs Control Accuracy Plots over 10 Epochs")
    plt.plot(x, bn_acc_hist, color="orange", label="BN Training Accuracy")
    plt.plot(x, bn_val_acc_hist, "--", color="orange", label="BN Validation Accuracy")
    plt.plot(x, no_acc_hist, color="blue", label="Control Training Accuracy")
    plt.plot(x, no_val_acc_hist, "--", color="blue", label="Control Validation Accuracy")
    plt.legend()
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")

    plt.ylim([0.0, 1.0])
    plt.xlim([0, bn_acc_hist.shape[0] +1])
    plt.savefig("AccHist.png")


    plt.figure().clear()
    plt.title("BN vs Control Loss Plots over 10 Epochs")
    plt.plot(x, bn_loss_hist, color="orange", label="BN Training Loss")
    plt.plot(x, bn_val_loss_hist, "--", color="orange", label="BN Validation Loss")
    plt.plot(x, no_loss_hist, color="blue", label="Control Training Loss")
    plt.plot(x, no_val_loss_hist, "--", color="blue", label="Control Validation Loss")
    plt.legend()
    plt.ylabel("Loss")
    plt.xlabel("Epochs")

    plt.ylim([0.0, 0.8])
    plt.xlim([0, bn_acc_hist.shape[0] +1])
    plt.savefig("LossHist.png")

    

if __name__ == "__main__":
    # X, Y, class_to_idx = import_dataset()
    # print(X.shape, Y.shape, class_to_idx)

    # print(Y)

    # dataset = pd.read_csv("HIGGS_reduced.csv")
    # l = dataset.shape[0]
    # new_dataset = dataset.iloc[:int(l*10/10), 1:]
    # new_dataset.to_csv("HIGGS_reduced.csv")
    # pass

    create_plots()

