import numpy as np
from sklearn.model_selection import train_test_split
from copy import deepcopy

from utils import *
from batchnorm import *
import pandas as pd

class Model:
    def __init__(self, input_dims, output_dims, alpha=0.01, decay=1e-8, N_hidden=10):

        N_h = N_hidden

        self.W1 = np.random.randn(input_dims, N_h) * np.sqrt(2/(input_dims + N_h))
        self.W2 = np.random.randn(N_h, N_h) * np.sqrt(2/(N_h + N_h))
        self.W3 = np.random.randn(N_h, N_h) * np.sqrt(2/(N_h + N_h))
        self.W4 = np.random.randn(N_h, N_h) * np.sqrt(2/(N_h + N_h))
        self.W5 = np.random.randn(N_h, output_dims)* np.sqrt(2/(N_h + output_dims))

        self.b1 = np.zeros((1, N_h))
        self.b2 = np.zeros((1, N_h))
        self.b3 = np.zeros((1, N_h))
        self.b4 = np.zeros((1, N_h))
        self.b5 = np.zeros((1, output_dims))


        self.dW1 = np.zeros(self.W1.shape)
        self.dW2 = np.zeros(self.W2.shape)
        self.dW3 = np.zeros(self.W3.shape)
        self.dW4 = np.zeros(self.W4.shape)
        self.dW5 = np.zeros(self.W5.shape)

        self.db1 = np.zeros(self.b1.shape)
        self.db2 = np.zeros(self.b2.shape)
        self.db3 = np.zeros(self.b3.shape)
        self.db4 = np.zeros(self.b4.shape)
        self.db5 = np.zeros(self.b5.shape)

        self.alpha = alpha
        self.decay = decay

        self.acc_hist = []
        self.v_acc_hist = []
        self.loss_hist = []
        self.v_loss_hist = []

        self.best_val_acc = 0
        self.best_state = None
        self.best_epoch = 0

        self.train = False

    def forward(self, X):
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = ReLU(Z1)

        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = ReLU(Z2)

        Z3 = np.dot(A2, self.W3) + self.b3
        A3 = ReLU(Z3)

        Z4 = np.dot(A3, self.W4) + self.b4
        A4 = ReLU(Z4)

        Z5 = np.dot(A4, self.W5) + self.b5
        A5 = Sigmoid(Z5)

        if self.train:
            p = {"X":X, "Z1":Z1, "A1":A1, 
                        "Z2":Z2, "A2":A2, 
                        "Z3":Z3, "A3":A3,
                        "Z4":Z4, "A4":A4,
                        "Z5":Z5, "A5":A5}
            return p

        return A5

    def backward(self, layers, Y):
        X = layers["X"]
        Z1 = layers["Z1"]
        A1 = layers["A1"]

        Z2 = layers["Z2"]
        A2 = layers["A2"]

        Z3 = layers["Z3"]
        A3 = layers["A3"]

        Z4 = layers["Z4"]
        A4 = layers["A4"]       
        
        Z5 = layers["Z5"]
        A5 = layers["A5"]

        M = Y.shape[0]



        dZ5 = A5 - Y
        self.dW5 = np.dot(A4.T, dZ5)/M
        self.db5 = np.sum(dZ5, axis=0)/M

        dA4 = np.dot(dZ5, self.W5.T)
        dZ4 = dA4 * (Z4 > 0).astype(np.float32)
        self.dW4 = np.dot(A3.T, dZ4)/M
        self.db4 = np.sum(dZ4, axis=0)/M

        dA3 = np.dot(dZ4, self.W4.T)
        dZ3 = dA3 * (Z3 > 0).astype(np.float32)
        self.dW3 = np.dot(A2.T, dZ3)/M
        self.db3 = np.sum(dZ3, axis=0)/M


        dA2 = np.dot(dZ3, self.W3.T)
        dZ2 = dA2 * (Z2 > 0).astype(np.float32)
        self.dW2 = np.dot(A1.T, dZ2)/M
        self.db2 = np.sum(dZ2, axis=0)/M

        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * (Z1 > 0).astype(np.float32)
        self.dW1 = np.dot(X.T, dZ1)/M
        self.db1 = np.sum(dZ1, axis=0)/M

    def step(self):

        self.W1 -= self.alpha * (self.dW1 - self.decay * self.W1)
        self.W2 -= self.alpha * (self.dW2 - self.decay * self.W2)
        self.W3 -= self.alpha * (self.dW3 - self.decay * self.W3)
        self.W4 -= self.alpha * (self.dW4 - self.decay * self.W4)
        self.W5 -= self.alpha * (self.dW5 - self.decay * self.W5)

        self.b1 -= self.alpha * self.db1
        self.b2 -= self.alpha * self.db2
        self.b3 -= self.alpha * self.db3 
        self.b4 -= self.alpha * self.db4 
        self.b5 -= self.alpha * self.db5 


    def fit(self, X, Y, epochs=1, batch_size=32, X_val=None, Y_val=None):

        M_train = X.shape[0]
        steps_per_epoch = int(np.ceil(M_train / batch_size))


        # Keeps a running average of the validation accuracies
        vaa = 0

        for epoch in range(epochs):
            self.set_train(True)
            avg_loss = 0
            avg_accuracy = 0
            for step in range(steps_per_epoch):
                X_b = X[step*batch_size:(step + 1)*batch_size, :]
                Y_b = Y[step*batch_size:(step + 1)*batch_size, :]

                lays = self.forward(X_b)
                Y_hat = lays["A5"]


                loss = BCE(Y_hat, Y_b)
                acc = accuracy_binary(Y_hat, Y_b)[0]

                print(f"\repoch: {epoch + 1}/{epochs}, step: {step + 1}/{steps_per_epoch}, loss: {loss:.4f}, accuracy: {acc:.4f}", end="")

                self.backward(lays, Y_b)
                self.step()

                avg_loss = 0.8*avg_loss + 0.2*loss
                avg_accuracy = 0.8*avg_accuracy + 0.2*acc


            if X_val is not None and Y_val is not None:
                M_val = X_val.shape[0]
                val_steps = int(np.ceil(M_val / batch_size))
                self.set_train(False)

                val_loss = 0
                val_acc = 0

                for step in range(val_steps):
                    X_val_b = X_val[step*batch_size:(step + 1)*batch_size, :]
                    Y_val_b = Y_val[step*batch_size:(step + 1)*batch_size, :]

                    Y_val_hat = self.forward(X_val_b)
                    
                    val_loss += BCE(Y_val_hat, Y_val_b)
                    val_acc += accuracy_binary(Y_val_hat, Y_val_b)[0]
                val_loss /= val_steps
                val_acc /= val_steps

                print(f", val_loss: {val_loss:.4f}, val_accuracy: {val_acc:.4f}")

                # keeps a running average validation accuracy of the last 5 epochs
                vaa = 0.8 * vaa + 0.2 * val_acc

                # keeps track of the history of validation metrics
                self.v_loss_hist.append(val_loss)
                self.v_acc_hist.append(val_acc)

                # keeps track of the best performing state
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_epoch = epoch + 1
                    self.best_state = deepcopy(self)
            else:
                print("")
            self.loss_hist.append(avg_loss)
            self.acc_hist.append(avg_accuracy)

        

        self.set_train(False)
        return self.loss_hist, self.acc_hist, self.v_loss_hist, self.v_acc_hist, vaa



    def set_train(self, train):
        self.train = train





            



    


    

def CCE(Y_hat, Y):

    return np.sum(np.sum(-Y * np.log(Y_hat*(1-2e-9) + 1e-9), axis=1))/Y.shape[0]

def BCE(Y_hat, Y):
    return np.sum(-Y * np.log(Y_hat*(1-2e-9) + 1e-9) - (1-Y) * np.log((1-Y_hat)*(1-2e-9) + 1e-9))/Y.shape[0]


def ReLU(z):
    return np.maximum(0, z)

def Softmax(z):
    znorm = z - np.amax(z, axis=1, keepdims=True)
    return np.exp(znorm) / np.sum(np.exp(znorm), axis=1, keepdims=True)

def Sigmoid(z):
    return 1 / (1 + np.exp(-z))


def accuracy(Y_hat, Y):
    Y_hat_lab = np.argmax(Y_hat, axis=1)
    Y_lab = np.argmax(Y, axis=1)
    return np.sum(Y_hat_lab == Y_lab) / Y.shape[0]

def accuracy_binary(Y_hat, Y):
    Y_hat = (Y_hat > 0.5).astype(np.int32)
    Y = Y.astype(np.int32)
    return np.sum(Y == Y_hat, axis=0)/Y.shape[0]


        





def h_param_opt(X_train, Y_train, X_val, Y_val, evolution_steps=10, h_param_steps=10):

    decay_l, decay_h = (-7.39, -7.55)
    hidden_l, hidden_h = (1000, 1000)
    lr_l, lr_h = (-0.21, -0.24)


    for evostep in range(evolution_steps):

        print(f"Beginning Hyperparameter Search Generation {evostep+1}/{evolution_steps}")
        
        decay = 10**np.random.uniform(decay_l, decay_h, h_param_steps)
        hidden = np.random.uniform(hidden_l, hidden_h, h_param_steps).astype(np.int32)
        alpha = 10**np.random.uniform(lr_l, lr_h, h_param_steps)
        
        # validation average accuracies
        vaas = []


        for h_step in range(h_param_steps):
            print(f"Model {h_step + 1}: L2: {decay[h_step]}, learning rate: {alpha[h_step]}, hidden units: {hidden[h_step]}")
            model = Model(input_dims=28, output_dims=1, alpha=alpha[h_step], decay=decay[h_step], N_hidden=hidden[h_step])
            _, _, _, _, vaa = model.fit(X_train, Y_train, X_val=X_val, Y_val=Y_val, epochs=20, batch_size=1024)

            vaas.append(vaa)
            print("\n")


        placeholder = np.array([vaas, np.log10(decay), hidden, np.log10(alpha)]).T
        df = pd.DataFrame(data=placeholder, columns=["val_acc", "L2", "hidden", "LR"])
        df = df.sort_values(by=["val_acc"], axis=0, ascending=False)
        top = df.iloc[:h_param_steps//3, :]
        means = top.mean(axis=0)
        stddev = top.std(axis=0)

        # hidden_l = int(means["hidden"] - 1.5*stddev["hidden"])
        # hidden_h = int(means["hidden"] + 1.5*stddev["hidden"])

        decay_l = means["L2"] + 1.5*stddev["L2"]
        decay_h = means["L2"] - 1.5*stddev["L2"]

        lr_l = means["LR"] + 1.5*stddev["LR"]
        lr_h = means["LR"] - 1.5*stddev["LR"]

        print(df)
        
        print(f"Hidden bounds: [{hidden_l}, {hidden_h}]")
        print(f"L2 bounds: [{decay_l}, {decay_h}]")
        print(f"Learning rate bounds: [{lr_l}, {lr_h}]")


        






if __name__ == "__main__":

    model = Model(28, 1, alpha=10**-0.21, decay=10**-7.4, N_hidden=1000)

    X, Y = load_higgs_dataset()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.15)

    print(X.shape, Y.shape)

    # hyperparameter search. May take all day, so comment this out 
    # if you simply want to train the model for 10 epochs
    h_param_opt(X_train, Y_train, X_val, Y_val, 10, 10)

    
    loss_history, acc_history, val_loss_history, val_acc_history, vaa = model.fit(X_train, Y_train, X_val=X_val, Y_val=Y_val, epochs=10, batch_size=1024)

    np.save("arrays/no_loss_hist", loss_history)
    np.save("arrays/no_acc_hist", acc_history)
    np.save("arrays/no_val_loss_hist", val_loss_history)
    np.save("arrays/no_val_acc_hist", val_acc_history)

    Y_pred = model.forward(X_test)
    acc = accuracy_binary(Y_pred, Y_test)
    print(f"Accuracy on test set: {acc[0]:.4f}")

    Y_pred_opt = model.best_state.forward(X_test)
    acc_opt = accuracy_binary(Y_pred_opt, Y_test)
    print(f"Optimal accuracy of epoch {model.best_epoch} on test set: {acc_opt[0]:.4f}")