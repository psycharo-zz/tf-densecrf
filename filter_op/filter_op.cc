#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

// #include "tensorflow/core/kernels/fill_functor.h"
// TODO: for now we assume GOOGLE_CUDA=1 by default
//#if GOOGLE_CUDA
// #include "filter_op_gpu.h"
// #include "tensorflow/core/platform/stream_executor.h"
//#endif  // GOOGLE_CUDA

#include "permutohedral.h"


REGISTER_OP("PermutoInit")
.Input("features: float32")
.Output("offsets: int32")
.Output("weights: float32")
.Output("neighbours: int32");

REGISTER_OP("PermutoCompute")
.Input("input: float32")
.Input("offsets: int32")
.Input("weights: float32")
.Input("neighbours: int32")
.Output("output: float32");// TODO: add shape attribute?


using tensorflow::int32;

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

void PermutohedralComputeKernelLauncher(const float* input,
                                        const int32* offsets,
                                        const float* weights,
                                        const int32* neighbours,
                                        int n_values,
                                        int n_points,
                                        int n_features,
                                        int n_vertices,
                                        bool reverse,
                                        float* output);


/**
 * Op that does initialization of weights, offsets and neighbours
 * NOTE: this a CPU-only kernel, GPU needs a hash-table implementation which
 * can be tedious, but it can be done.
 * See e.g. here http://graphics.stanford.edu/papers/permutohedral/
 */
template <typename T>
class PermutoInitOp : public OpKernel {
public:
  explicit PermutoInitOp(OpKernelConstruction* context) : OpKernel(context) {
  }

  void Compute(OpKernelContext* context) {
    const Tensor& features = context->input(0);

    // TODO: see if dynamic allocation makes a difference
    const int num_points = features.shape().dim_size(0);
    const int num_dims = features.shape().dim_size(1);

    // TODO: 2D or 3D?
    OP_REQUIRES(context,
                TensorShapeUtils::IsMatrix(features.shape()),
                errors::InvalidArgument("features is not a matrix"));

    Tensor* offsets = nullptr;
    Tensor* weights = nullptr;
    Tensor* neighbours = nullptr;

    // TODO: this depends on whether it is sse or not

    // TODO: this is done dynamically, e.g. split init() into several functions
    int32 *nbs;
    int num_vertices;


    OP_REQUIRES_OK(context,
                   context->allocate_output(0,
                                            TensorShape({num_points*16, num_dims+1}),
                                            &offsets));
    OP_REQUIRES_OK(context,
                   context->allocate_output(1,
                                            TensorShape({num_points*16, num_dims+1}),
                                            &weights));

    permutohedral::init_sse(features.flat<T>().data(), num_points, num_dims,
                            offsets->flat<int32>().data(), weights->flat<T>().data(),
                            nbs, num_vertices);


    OP_REQUIRES_OK(context,
                   context->allocate_output(2,
                                            TensorShape({num_vertices, num_dims+1, 2}),
                                            &neighbours));

    // TODO: copying stuff
    memcpy(neighbours->flat<int32>().data(), nbs, sizeof(int32) * num_vertices*(num_dims+1)*2);
    delete [] nbs;
  }
};


/**
 * This op does the actual high-dimensional filtering given pre-computed
 * lattice
 * TODO: this is CPU-only, merge it into a single op
 */
template <typename T>
class PermutoComputeCPUOp : public OpKernel {
public:
  explicit PermutoComputeCPUOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) {
    const Tensor& input = context->input(0);
    const Tensor& offsets = context->input(1);
    const Tensor& weights = context->input(2);
    const Tensor& neighbours = context->input(3);

    // TODO: add more input checks
    OP_REQUIRES(context,
                TensorShapeUtils::IsMatrix(input.shape()),
                errors::InvalidArgument("input is not a matrix"));

    Tensor* output;

    int num_values = input.shape().dim_size(1);
    int num_points = input.shape().dim_size(0);
    int num_features = neighbours.shape().dim_size(1)-1;
    int num_vertices = neighbours.shape().dim_size(0);

    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &output));

    permutohedral::compute_sse(input.flat<T>().data(),
                               offsets.flat<int32>().data(),
                               weights.flat<T>().data(),
                               neighbours.flat<int32>().data(),
                               num_values, num_points, num_features, num_vertices,
                               output->flat<T>().data());
  }
};


// TODO: change this ambiguous name, it should be a template of PermutoCompute
template <typename T>
class PermutoComputeGPUOp : public OpKernel {
public:
  explicit PermutoComputeGPUOp(OpKernelConstruction* context) : OpKernel(context) {
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& offsets = context->input(1);
    const Tensor& weights = context->input(2);
    const Tensor& neighbours = context->input(3);

    // TODO: add more input checks
    OP_REQUIRES(context,
                TensorShapeUtils::IsMatrix(input.shape()),
                errors::InvalidArgument("input is not a matrix"));

    Tensor* output;

    int n_values = input.shape().dim_size(1);
    int n_points = input.shape().dim_size(0);
    int n_features = neighbours.shape().dim_size(1)-1;
    int n_vertices = neighbours.shape().dim_size(0);

    // TODO: also create values here?
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &output));

    PermutohedralComputeKernelLauncher(input.flat<T>().data(),
                                       offsets.flat<int32>().data(),
                                       weights.flat<T>().data(),
                                       neighbours.flat<int32>().data(),
                                       n_values,
                                       n_points,
                                       n_features,
                                       n_vertices,
                                       false,
                                       output->flat<T>().data());

    LOG(INFO) << "just finished";
  }
};


// TODO: we also need to register shape
REGISTER_KERNEL_BUILDER(Name("PermutoInit")
                        .Device(DEVICE_CPU),
                        PermutoInitOp<float>);

REGISTER_KERNEL_BUILDER(Name("PermutoCompute")
                        .Device(DEVICE_CPU),
                        PermutoComputeCPUOp<float>);

REGISTER_KERNEL_BUILDER(Name("PermutoCompute")
                        .Device(DEVICE_GPU),
                        PermutoComputeGPUOp<float>);

} // namespace tensorflow
