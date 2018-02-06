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

# what is below is a pure TF implementation of compute/filter method    
def ph_splat(inputs, offsets, weights, nbs):
    N, C = inputs.shape
    F = weights.shape[1] - 1
    M = tf.shape(nbs)[0]
    weighted_inputs = tf.matmul(weights[:N,:,tf.newaxis],
                                inputs[:N,tf.newaxis,:])
    weighted_inputs = tf.reshape(weighted_inputs, [-1, C])
    idxs = tf.reshape(offsets[:N,:F+1], [-1,1])+1
    # TODO: the only thing is the unknown shape of M?
    # NOTE: the docs say the update is not deterministic, 
    # but it seems to work
    return tf.scatter_nd(idxs, weighted_inputs, [M+2, C])

def ph_blur(inputs, values_in, offsets, weights, nbs):
    # TODO: we can parameterize all this !
    def _blur_iter(prev, nbs):
        n1 = tf.gather(prev, nbs[:,0]+1)
        n2 = tf.gather(prev, nbs[:,1]+1)
        return prev + 0.5 * tf.pad(n1 + n2, [[1,1], [0,0]])
    return tf.foldl(_blur_iter, 
                    tf.transpose(nbs, [1, 0, 2]), 
                    values_in)

def ph_slice(inputs, values_in, offsets, weights, nbs):
    N, C = inputs.shape
    F = int(weights.shape[1])-1
    alpha = 1.0 / (1.0 + 2.0**(-F))
    idxs = tf.reshape(offsets[:N,:F+1], [-1,])+1
    w = weights[:N,:,tf.newaxis]
    v = tf.reshape(tf.gather(values_in, idxs), [N, F+1, C])
    return tf.reduce_sum(alpha * w * v, axis=1)    

def ph_filter(inputs, lattice):
    values = ph_splat(inputs, *lattice)
    values = ph_blur(inputs, values, *lattice)
    return ph_slice(inputs, values, *lattice)    