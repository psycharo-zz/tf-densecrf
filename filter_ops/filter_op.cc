#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "permutohedral.h"

using namespace tensorflow;
using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("PermutoInit")
.Input("features: float32")
.Output("offsets: int32")
.Output("weights: float32")
.Output("neighbours: int32")
.SetShapeFn([](InferenceContext* c) {
    ShapeHandle features;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &features));
    DimensionHandle num_dims_out;
    TF_RETURN_IF_ERROR(c->Add(c->Dim(features, 1), c->MakeDim(1), &num_dims_out));
    DimensionHandle num_points_out;
    TF_RETURN_IF_ERROR(c->Add(c->Dim(features, 0), c->MakeDim(16), &num_points_out));

    c->set_output(0, c->MakeShape({num_points_out, num_dims_out}));
    c->set_output(1, c->MakeShape({num_points_out, num_dims_out}));
    c->set_output(2, c->MakeShape({c->UnknownDim(), num_dims_out, c->MakeDim(2)}));
    return Status::OK();
  });


REGISTER_OP("PermutoCompute")
.Input("input: float32")
.Input("offsets: int32")
.Input("weights: float32")
.Input("neighbours: int32")
.Output("output: float32")
.Attr("reverse: bool = false")
.SetShapeFn([](InferenceContext* c) {
    ShapeHandle unused;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &unused));
    c->set_output(0, c->input(0));
    return Status::OK();
  });

REGISTER_OP("Splat")
.Input("input: float32")
.Input("offsets: int32")
.Input("weights: float32")
.Input("neighbours: int32")
.Output("output: float32");

REGISTER_OP("Blur")
.Input("input: float32")
.Input("values: float32")
.Input("offsets: int32")
.Input("weights: float32")
.Input("neighbours: int32")
.Output("output: float32");


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
  explicit PermutoInitOp(OpKernelConstruction* context)
    : OpKernel(context) {}

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

    // TODO: this is done dynamically, e.g. split init() into several functions
    int32 *nbs;
    int num_vertices;

    // NOTE: not sure why +16, sse?
    OP_REQUIRES_OK(context,
                   context->allocate_output(0,
                                            TensorShape({num_points+16, num_dims+1}),
                                            &offsets));
    OP_REQUIRES_OK(context,
                   context->allocate_output(1,
                                            TensorShape({num_points+16, num_dims+1}),
                                            &weights));

    // NOTE: init() does not seem to work well
    permutohedral::init_sse(features.flat<T>().data(), num_points, num_dims,
                            offsets->flat<int32>().data(), weights->flat<T>().data(),
                            nbs, num_vertices);

    OP_REQUIRES_OK(context,
                   context->allocate_output(2,
                                            TensorShape({num_vertices, num_dims+1, 2}),
                                            &neighbours));

    // TODO: is copying necessary?
    memcpy(neighbours->flat<int32>().data(), nbs, sizeof(int32) * num_vertices*(num_dims+1)*2);
    delete [] nbs;
  }
};



/**
 * This op does the actual high-dimensional filtering given pre-computed lattice:
 * CPU-SSE implementation
 */
template <typename T>
class PermutoComputeCPUOp : public OpKernel {
public:
  explicit PermutoComputeCPUOp(OpKernelConstruction* context)
    : OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("reverse", &reverse_));
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

    int n_values = input.shape().dim_size(1);
    int n_points = input.shape().dim_size(0);
    int n_features = neighbours.shape().dim_size(1)-1;
    int n_vertices = neighbours.shape().dim_size(0);

    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &output));

    permutohedral::compute_sse(input.flat<T>().data(),
                               offsets.flat<int32>().data(),
                               weights.flat<T>().data(),
                               neighbours.flat<int32>().data(),
                               n_values, n_points, n_features, n_vertices,
                               output->flat<T>().data(),
                               reverse_);
  }
private:
  bool reverse_;
};



template <typename T>
class SplatCPUOp : public OpKernel {
public:
  explicit SplatCPUOp(OpKernelConstruction* context)
    : OpKernel(context) {}


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

    int V = input.shape().dim_size(1);
    int N = input.shape().dim_size(0);
    int F = neighbours.shape().dim_size(1)-1;
    int M = neighbours.shape().dim_size(0);

    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape{(M+2), V}, &output));


    auto* input_ptr = input.flat<T>().data();
    auto* offsets_ptr = offsets.flat<int32>().data();
    auto* weights_ptr = weights.flat<T>().data();
    auto* values_ptr = output->flat<T>().data();

    // splatting
    for (int i = 0; i < (M+2)*V; i++)
      values_ptr[i] = 0;

    for (int i = 0; i < N; i++) {
      for (int j = 0; j <= F; j++) {
        int o = offsets_ptr[i*(F+1)+j]+1;
        float w = weights_ptr[i*(F+1)+j];
        for (int k=0; k < V; k++)
          values_ptr[o*V+k] += w * input_ptr[i*V+k];
      }
    }
  }
};

template <typename T>
class BlurCPUOp : public OpKernel {
public:
  explicit BlurCPUOp(OpKernelConstruction* context)
    : OpKernel(context) {}


  void Compute(OpKernelContext* context) {
    const Tensor& input = context->input(0);
    const Tensor& values_in = context->input(1);
    const Tensor& offsets = context->input(2);
    const Tensor& weights = context->input(3);
    const Tensor& neighbours = context->input(4);


    // TODO: add more input checks
    OP_REQUIRES(context,
                TensorShapeUtils::IsMatrix(input.shape()),
                errors::InvalidArgument("input is not a matrix"));

    Tensor* output;

    int V = input.shape().dim_size(1);
    int N = input.shape().dim_size(0);
    int F = neighbours.shape().dim_size(1)-1;
    int M = neighbours.shape().dim_size(0);

    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape{(M+2), V}, &output));

    auto* input_ptr = input.flat<T>().data();
    auto* offsets_ptr = offsets.flat<int32>().data();
    auto* weights_ptr = weights.flat<T>().data();
    auto* nbs_ptr = neighbours.flat<int32>().data();

    auto* values_in_ptr = values_in.flat<T>().data();
    auto* values_ptr = output->flat<T>().data();
    std::copy(values_in_ptr, values_in_ptr + (M+2)*V, values_ptr);

    std::vector<float> new_values((M+2)*V, 0.0);
    auto* new_values_ptr = new_values.data();

    // blurring
    for (int j = 0; j <= F; ++j) {
      for (int i = 0; i < M; i++) {
        float* old_val = values_ptr + (i+1) * V;
        float* new_val = new_values_ptr + (i+1) * V;

        int n1 = nbs_ptr[(i*(F+1)+j)*2]+1;
        int n2 = nbs_ptr[(i*(F+1)+j)*2+1]+1;
        float* n1_val = values_ptr + n1 * V;
        float* n2_val = values_ptr + n2 * V;
        for (int k = 0; k < V; k++)
          new_val[k] = old_val[k] + 0.5 * (n1_val[k] + n2_val[k]);
      }
      std::swap(values_ptr, new_values_ptr);
    }
    // saving to the output
    std::copy(values_ptr, values_ptr + (M+2)*V, output->flat<T>().data());
  }
};




/**
 * High-dimensional filtering on top given the lattice: GPU implementation
 */
template <typename T>
class PermutoComputeGPUOp : public OpKernel {
public:
  explicit PermutoComputeGPUOp(OpKernelConstruction* context)
    : OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("reverse", &reverse_));
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
                                       n_values, n_points, n_features, n_vertices,
                                       reverse_,
                                       output->flat<T>().data());
  }
private:
  bool reverse_;
};

REGISTER_KERNEL_BUILDER(Name("PermutoInit")
                        .Device(DEVICE_CPU),
                        PermutoInitOp<float>);

REGISTER_KERNEL_BUILDER(Name("PermutoCompute")
                        .Device(DEVICE_CPU),
                        PermutoComputeCPUOp<float>);

REGISTER_KERNEL_BUILDER(Name("PermutoCompute")
                        .Device(DEVICE_GPU),
                        PermutoComputeGPUOp<float>);

REGISTER_KERNEL_BUILDER(Name("Splat")
                        .Device(DEVICE_CPU),
                        SplatCPUOp<float>);

REGISTER_KERNEL_BUILDER(Name("Blur")
                        .Device(DEVICE_CPU),
                        BlurCPUOp<float>);



} // namespace tensorflow
