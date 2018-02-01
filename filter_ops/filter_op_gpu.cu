#define EIGEN_USE_GPU

#include <algorithm>

/* #include "tensorflow/core/framework/register_types.h" */

// these are necessary (!)
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

#define BLOCK_SIZE 8

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

__global__ void PermutohedralSplatKernel(const int N,
                                         const int F,
                                         const int V,
                                         const float* input,
                                         const int32* offsets,
                                         const float* weights,
                                         float* values) {

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int k = blockDim.y * blockIdx.y + threadIdx.y;
  int f = blockDim.z * blockIdx.z + threadIdx.z;

  if (i >= N || k >= V || f > F)
    return;

  extern __shared__ float s_input[];
  int sidx = threadIdx.y * blockDim.x + threadIdx.x;
  if (threadIdx.z == 0)
    s_input[sidx] = input[i*V+k];
  __syncthreads();

  int o = offsets[i*(F+1)+f]+1;
  float w = weights[i*(F+1)+f];
  atomicAdd(values + (o*V+k), w * s_input[sidx]);
}


__global__ void PermutohedralBlurKernel(int j, int M, int F, int V,
                                        const int32* neighbours,
                                        const float* values,
                                        float* new_values) {

  int tix = threadIdx.x;
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int k = blockDim.y * blockIdx.y + threadIdx.y;

  if (i >= M || k >= V)
    return;

  extern __shared__ int s_n[];
  if (threadIdx.y == 0) {
    s_n[2*tix] = neighbours[(i*(F+1)+j)*2]+1;
    s_n[2*tix+1] = neighbours[(i*(F+1)+j)*2+1]+1;
  }
  __syncthreads();

  const float* n1_val = values + s_n[2*tix] * V;
  const float* n2_val = values + s_n[2*tix+1] * V;

  new_values[(i+1)*V+k] = values[(i+1)*V+k] + 0.5 * (n1_val[k] + n2_val[k]);
}

__global__ void PermutohedralSliceKernel(int N, int F, int V,
                                         const int32* offsets,
                                         const float* weights,
                                         const float* values,
                                         float alpha,
                                         float* output) {

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int k = blockDim.y * blockIdx.y + threadIdx.y;

  if (i >= N || k >= V)
    return;

  float wsum = 0.0;
  for (int f = 0; f <= F; ++f) {
    int o = offsets[i*(F+1)+f]+1;
    float w = weights[i*(F+1)+f];
    wsum += w * values[o*V+k] * alpha;
  }
  output[i*V+k] = wsum;
}

__global__ void PermutoHedralSliceKernelShared(int N, int F, int V,
                                               const int32* offsets,
                                               const float* weights,
                                               const float* values,
                                               float alpha,
                                               float* output) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int k = blockDim.y * blockIdx.y + threadIdx.y;
  int f = blockDim.z * blockIdx.z + threadIdx.z;

  if (i >= N || k >= V || f > F)
    return;

  __shared__ int s_offsets[BLOCK_SIZE*BLOCK_SIZE];
  __shared__ float s_weights[BLOCK_SIZE*BLOCK_SIZE];
  //cint sidx = threadIdx.z * blockDim.x + threadIdx.x;
  int sidx = threadIdx.z + blockDim.z * threadIdx.x;
  if (threadIdx.y == 0) {
    s_offsets[sidx] = offsets[i*(F+1)+f]+1;
    s_weights[sidx] = weights[i*(F+1)+f];
  }
  if (threadIdx.z == 0)
    output[i*V+k] = 0.0;
  __syncthreads();

  int o = s_offsets[sidx];
  float w = s_weights[sidx];
  atomicAdd(output + i*V+k, w * values[o*V+k] * alpha);
}

// for now, tensors are allocated by the op or manually
void PermutohedralComputeKernelLauncher(const float* input,
                                        const int32* offsets,
                                        const float* weights,
                                        const int32* neighbours,
                                        int n_values,
                                        int n_points,
                                        int n_features,
                                        int n_vertices,
                                        bool reverse,
                                        float* output) {
  int V = n_values;
  int N = n_points;
  int F = n_features;
  int M = n_vertices;

  // TODO: this can be allocated only once during the initialization (in the op itself)
  float *values;
  float *new_values;
  cudaMalloc((void**)&(values), (M+2)*V*sizeof(float));
  cudaMemset(values, 0, (M+2)*V*sizeof(float));
  cudaMalloc((void**)&(new_values), (M+2)*V*sizeof(float));
  cudaMemset(new_values, 0, (M+2)*V*sizeof(float));

  // TODO: (!!!) add some kind of assertion for the # of classes
  // and feature dimensionality, for now it is assumed larger then # of
  const int32 bsize = BLOCK_SIZE;

  // splatting
  dim3 blocks_splat((N-1) / bsize + 1,
                    (V-1) / bsize + 1,
                    F / bsize + 1);
  dim3 threads_splat(bsize, bsize, bsize);
  PermutohedralSplatKernel<<<blocks_splat,threads_splat,bsize*bsize>>>(
      N, F, V, input, offsets, weights, values);

  // blurring
  /* dim3 blocks_blur((M-1) / (4*bsize) + 1, */
  /*                  (V-1) / (bsize) + 1, */
  /*                  1); */
  /* dim3 threads_blur(4*bsize, bsize, 1); */
  dim3 blocks_blur((M-1) / bsize + 1,
                   (V-1) / bsize + 1,
                   1);
  dim3 threads_blur(bsize, bsize, 1);
  for (int f = reverse ? F : 0; f <= F && f >= 0; reverse ? --f : ++f) {
    PermutohedralBlurKernel<<<blocks_blur,threads_blur, 2*4*bsize>>>(
        f, M, F, V, neighbours, values, new_values);
    std::swap(values, new_values);
  }

  // slicing, can be done fully parallel
  float alpha = 1.0f / (1.0f + powf(2, -F));

  // using shared memory (assuming bsize=2)
  /* dim3 blocks_slice((N-1) / bsize + 1, */
  /*                   (V-1) / bsize + 1, */
  /*                   F / bsize + 1); */
  /* dim3 threads_slice(bsize, bsize, bsize); */
  dim3 blocks_slice((N-1) / bsize + 1,
                    (V-1) / bsize + 1,
                    1);
  dim3 threads_slice(bsize, bsize, 1);
  PermutohedralSliceKernel<<<blocks_slice,threads_slice>>>(
      N, F, V, offsets, weights, values, alpha, output);

  // free tmp memory
  cudaFree(values);
  cudaFree(new_values);
}

} // namespace tensorflow
