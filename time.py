import main, main_no_bn
import numpy as np
from time import time



# Used to time inference time for models with and without batch norm
# Does not test training time (that was timed using a stopwatch)
if __name__ == "__main__":
    bn_model = main.Model(28, 1, N_hidden=1000)
    control = main_no_bn.Model(28, 1, N_hidden=1000)

    x = np.random.randn(1024, 28)*0.5

    bn_model.set_train(False)
    control.set_train(False)

    start = time()
    for i in range(100):
        bn_model.forward(x)
    stop = time()
    print(f"Time taken for BN Model: {stop-start}s")


    start = time()
    for i in range(100):
        control.forward(x)
    stop = time()
    print(f"Time taken for Control Model: {stop-start}s")
