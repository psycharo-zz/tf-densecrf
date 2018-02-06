BUILD=./build

all: $(BUILD)/filter_op.so

clean:
	rm -rf $(BUILD)

# NOTE: assuming conda 5.0+, and
CXX=$(HOME)/anaconda3/bin/x86_64-conda_cos6-linux-gnu-g++

# NOTE: CUDA_HOME should be set, make sure to set correct gpu-architecture flag
NV_INC?=$(CUDA_HOME)/include
NV_LIB?=$(CUDA_HOME)/lib64

TF_INC=$(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
# NOTE: switch to this when developing
# TF_INC=$(HOME)/anaconda3/lib/python3.6/site-packages/tensorflow/include
# TF_LIB=$(HOME)/anaconda3/lib/python3.6/site-packages/tensorflow

# note: this is necessary due to https://github.com/tensorflow/tensorflow/issues/15002
TF_SRC=$(HOME)/src/tensorflow
TF_CUDA_FLAGS=-I$(TF_SRC)/third_party/toolchains/gpus/cuda -I$(CUDA_HOME)/../

# NOTE: for older gcc it's probably necessary to remove -D_GLIBCXX_USE_CXX11_ABI=0
$(BUILD)/filter_op.so: $(BUILD)/filter_op.o $(BUILD)/filter_op_gpu.cu.o $(BUILD)/permutohedral.o
	$(CXX) -O3 -std=c++11 $^ -shared -o $@ -L$(TF_LIB) -ltensorflow_framework -L$(NV_LIB) -lcudart -fPIC -D_GLIBCXX_USE_CXX11_ABI=0

$(BUILD)/%.o: %.cc
	mkdir -p $(shell dirname $@)
	$(CXX) -O3 -std=c++11 -c $< -o $@ -I$(TF_INC) -I$(TF_INC)/external/nsync/public -DGOOGLE_CUDA=1 -fPIC -D_GLIBCXX_USE_CXX11_ABI=0

$(BUILD)/%.cu.o: %.cu
	mkdir -p $(shell dirname $@)
	nvcc -std=c++11 -c $< -o $@  -I$(TF_INC) -I$(TF_INC)/external/nsync/public $(TF_CUDA_FLAGS) -DGOOGLE_CUDA=1 --expt-relaxed-constexpr -Xcompiler -fPIC --gpu-architecture=sm_61
