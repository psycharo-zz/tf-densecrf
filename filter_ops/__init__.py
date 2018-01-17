import os
import tensorflow as tf

_filter_ops = tf.load_op_library(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                              'build/filter_op.so'))

permuto_init = _filter_ops.permuto_init
permuto_compute = _filter_ops.permuto_compute

@tf.RegisterGradient("PermutoCompute")
def _permuto_compute_grad(op, grad):
      lattice = op.inputs[1:]
      out = permuto_compute(grad, *lattice, reverse=True)
      return [out, None, None, None]
