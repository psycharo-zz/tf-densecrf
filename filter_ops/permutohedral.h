#ifndef PERMUTOHEDRAL_H_
#define PERMUTOHEDRAL_H_

#include "tensorflow/core/framework/types.h"

using tensorflow::int32;

namespace permutohedral {

void init_sse(const float* features, int num_points, int num_dims,
              int32* offsets, float* weights,
              int32*& neighbours, int &num_vertices);

/** compute the convolution of `input` with of size `n_values x n_values`
 * reverse - whether to perform blurring in reverse order
 * add - whether to overwrite or add to the output
 */
void compute_sse(const float* input,
                 const int32* offsets, const float* weights, const int32* neighbours,
                 int num_values, int num_points, int num_dims, int num_vertices,
                 float* output,
                 bool reverse = false,
                 bool add = false);

// TODO: make this take tensors as input
// computes offsets and barycentric weights
void init(const float* features, int num_points, int num_dims,
          int32* offsets, float* weights,
          int32*& neighbours, int &num_vertices);

// compute the convolution of `input` with of size `n_values x n_values`
void compute(const float* input,
             const int32* offsets, const float* weights, const int32* neighbours,
             int num_values, int num_points, int num_features, int num_vertices,
             float* output,
             bool reverse = false,
             bool add = false);

}

#endif // PERMUTOHEDRAL_H_
