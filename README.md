# tf-densecrf

DenseCRF in TensorFlow: high-dimensional filtering with GPU kernels.

First, build the op with `cd filter_op ; make `.
Make sure `CUDA_HOME` is set as well as appropriate flags in `filter_op/Makefile`.

Then, see the notebook `op_test.ipynb` for an example.

Tested on:
- Python 3.6
- TensorFlow 1.4.1
- Titan X Pascal

Currently only forward pass is implemented.
