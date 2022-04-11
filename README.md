# Batch-Norm

This is the code used in the paper *Exploring Batch Normalization* under the repo `zony249/BN-paper`. 
The two main files to look out for are `main.py` and `main_no_bn.py`. As the name may or may not suggest, `main.py` 
tests the batch-normalized model, whereas `main_no_bn.py` acts as a control with a standard, non-batch-normalized model.
Model architectures aside from batch normalization are identical.

## Install Dependencies

Dependencies used for this paper are very basic -- no deep learning libraries were used. To install:

```bash
pip install numpy pandas sklearn pillow matplotlib
```

## Running Models

To run the models, simply run either of the following:

```bash
# To run the Batch-Normalized model:
python3 main.py

# To run the Control model:
python3 main_no_bn.py
```

## Execution Time

Additionally, `time.py` tests the execution time between control and batch-normalized models. Execution time tested is inference time, 
where a batch of 1024 examples are repeatedly passed into the models for 100 iterations.
```bash
python3 time.py
```
