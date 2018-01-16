BUILD=./build

all: $(BUILD)/filter_op.so

clean:
	rm -rf $(BUILD)

# NOTE: CUDA_HOME should be set, make sure to set correct gpu-architecture flag
NV_INC?=$(CUDA_HOME)/include
NV_LIB?=$(CUDA_HOME)/lib64

TF_INC=/home/bagautdi/anaconda3/lib/python3.6/site-packages/tensorflow/include
TF_LIB=/home/bagautdi/anaconda3/lib/python3.6/site-packages/tensorflow
# NOTE: this is due to https://github.com/tensorflow/tensorflow/issues/15002
TF_CUDA_INC=/home/bagautdi/src/tensorflow/third_party/toolchains/gpus/cuda

$(BUILD)/filter_op.so: $(BUILD)/filter_op.o $(BUILD)/filter_op_gpu.cu.o $(BUILD)/permutohedral.o
	g++ -O3 -std=c++11 $^ -shared -o $@ -L$(TF_LIB) -ltensorflow_framework -L$(NV_LIB) -lcudart -lcuda -fPIC

$(BUILD)/%.o: %.cc
	mkdir -p $(shell dirname $@)
	g++ -O3 -std=c++11 -c $< -o $@ -I$(TF_INC) -I$(TF_INC)/external/nsync/public -DGOOGLE_CUDA=1 -fPIC

$(BUILD)/%.cu.o: %.cu
	mkdir -p $(shell dirname $@)
	nvcc -std=c++11 -c $< -o $@  -I$(TF_INC) -I$(TF_INC)/external/nsync/public -I$(TF_CUDA_INC) -I$(CUDA_HOME)/../ -DGOOGLE_CUDA=1 --expt-relaxed-constexpr -Xcompiler -fPIC --gpu-architecture=sm_61
