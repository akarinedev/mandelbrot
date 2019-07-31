CX_CPU := clang
CX_CPU_FLAGS := -std=c++11

CX_GPU := nvcc
CX_GPU_FLAGS :=
#--cuda-path=/usr/local/cuda --cuda-gpu-arch=sm_75 -L/usr/local/cuda/lib64/

LK := clang
LK_FLAGS :=

SRC_CPU := $(wildcard *.cpp)
SRC_GPU := $(wildcard *.cu)
OBJ_CPU := $(patsubst %.cpp, obj/cpu/%.o, $(SRC_CPU))
OBJ_GPU := $(patsubst %.cu, obj/gpu/%.o, $(SRC_GPU))
BIN := mandel

$(BIN): $(OBJ_CPU) $(OBJ_GPU)
	$(LK) $(LK_FLAGS) $^ -o $@

obj/cpu/%.o: %.cpp
	$(CX_CPU) $(CX_CPU_FLAGS) -c $< -o $@

obj/gpu/%.o: %.cu
	$(CX_GPU) $(CX_GPU_FLAGS) -c $< -o $@
