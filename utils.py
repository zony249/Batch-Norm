import numpy as np
from glob import iglob
from PIL import Image
import pickle
import os
import pandas as pd

def import_dataset(path="images"):
    
    count = 0 
    class_to_idx = {}

    try:
        X = np.load(path + "/x.npy")
        Y_hot = np.load(path + "/y.npy")
        with open(path + "/class_to_idx.pickle", 'rb') as fname:
            class_to_idx = pickle.load(fname)

        return X, Y_hot, class_to_idx
    except:
        os.system("rm images/*.npy images/class_to_idx.pickle")
        pass


    X = []
    Y = []
    for e in iglob(path + "/*"):
        classlabel = e.split("/")[-1]
        class_to_idx[classlabel] = count
        
        for images in iglob(e + "/*"):
            img = Image.open(images).resize((48, 48)).convert("RGB")
            img = np.array(img).reshape(-1)
            print(img.shape)
            X.append(img)
            Y.append(count)
            print(images)

        count += 1

    X = np.array(X)
    Y = np.array(Y).astype(int)
    Y_hot = np.zeros((Y.size, len(class_to_idx)))
    Y_hot[np.arange(Y.size), Y[:]] = 1

    shuffle = np.random.permutation(X.shape[0])

    X = X[shuffle]
    Y_hot = Y_hot[shuffle]


    np.save(path + "/x", X)
    np.save(path + "/y", Y_hot)
    with open(path + "/class_to_idx.pickle", 'wb') as fname:
        pickle.dump(class_to_idx, fname, protocol=pickle.HIGHEST_PROTOCOL)

    return (X, Y_hot, class_to_idx)


def load_higgs_dataset():
    dataset = pd.read_csv("HIGGS_reduced.csv", header=None)

    Y = dataset.iloc[:, 1:2].to_numpy().astype(np.float32)
    X = dataset.iloc[:, 2:].to_numpy().astype(np.float32)

    return X, Y

if __name__ == "__main__":
    # X, Y, class_to_idx = import_dataset()
    # print(X.shape, Y.shape, class_to_idx)

    # print(Y)

    dataset = pd.read_csv("HIGGS_reduced.csv")
    l = dataset.shape[0]
    new_dataset = dataset.iloc[:int(l*9/10), 1:]
    new_dataset.to_csv("HIGGS_reduced.csv")
    # pass


