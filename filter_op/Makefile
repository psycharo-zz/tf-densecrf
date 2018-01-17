BUILD=./build

all: $(BUILD)/filter_op.so

clean:
	rm -rf $(BUILD)

# NOTE: CUDA_HOME should be set, make sure to set correct gpu-architecture flag
NV_INC?=$(CUDA_HOME)/include
NV_LIB?=$(CUDA_HOME)/lib64

TF_INC=$(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

# NOTE: this is necessary due to https://github.com/tensorflow/tensorflow/issues/15002
TF_SRC=/home/bagautdi/src/tensorflow
TF_CUDA_FLAGS=-I$(TF_SRC)/third_party/toolchains/gpus/cuda -I$(CUDA_HOME)/../

$(BUILD)/filter_op.so: $(BUILD)/filter_op.o $(BUILD)/filter_op_gpu.cu.o $(BUILD)/permutohedral.o
	g++ -O3 -std=c++11 $^ -shared -o $@ -L$(TF_LIB) -ltensorflow_framework -L$(NV_LIB) -lcudart -lcuda -fPIC

$(BUILD)/%.o: %.cc
	mkdir -p $(shell dirname $@)
	g++ -O3 -std=c++11 -c $< -o $@ -I$(TF_INC) -I$(TF_INC)/external/nsync/public -DGOOGLE_CUDA=1 -fPIC

$(BUILD)/%.cu.o: %.cu
	mkdir -p $(shell dirname $@)
	nvcc -std=c++11 -c $< -o $@  -I$(TF_INC) -I$(TF_INC)/external/nsync/public $(TF_CUDA_FLAGS) -DGOOGLE_CUDA=1 --expt-relaxed-constexpr -Xcompiler -fPIC --gpu-architecture=sm_61
