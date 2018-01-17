# tf-densecrf

## DenseCRF in TensorFlow

Efficient mean-field inference and learning in dense CRFs.
This repository contains both CPU and GPU implementations of
high-dimensional filtering ops using permutohedral lattice.

First, build the op with `cd filter_ops ; make `.
Make sure `CUDA_HOME` is set as well as appropriate flags in `filter_ops/Makefile`.

Then, see the notebook `op_test.ipynb` for an example.

Tested on:
- Python 3.6
- TensorFlow 1.4.1
- Titan X Pascal
