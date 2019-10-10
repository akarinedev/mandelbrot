CUDA_DIR := /usr/local/cuda-10.0/
CUDA_ARCH := sm_75

CXX := clang++-8
CXX_FLAGS := -x cuda --cuda-path=$(CUDA_DIR) --cuda-gpu-arch=$(CUDA_ARCH) -fPIC -std=c++11 -D OPT_ABS

R_LNK := $(CUDA_DIR)bin/nvcc
R_LNK_FLAGS := -L$(CUDA_DIR)lib64/ -lcudart_static

I_LNK := $(CUDA_DIR)bin/nvcc
I_LNK_FLAGS := -L$(CUDA_DIR)lib64/ -lcudart_static -lncurses

SRC_DIR := src
SRC_EXT := cpp
OBJ_DIR := obj
OBJ_EXT := o

SRC := $(wildcard $(SRC_DIR)/*.$(SRC_EXT))
OBJ := $(patsubst $(SRC_DIR)/%.$(SRC_EXT), $(OBJ_DIR)/%.$(OBJ_EXT), $(SRC))

$(OBJ_DIR)/%.$(OBJ_EXT): $(SRC_DIR)/%.$(SRC_EXT)
	$(CXX) $(CXX_FLAGS) -c $< -o $@

render: obj/main.o obj/gpu.o
	$(R_LNK) $(R_LNK_FLAGS) $^ -o $@

interactive: obj/interactive.o obj/gpu.o
	$(I_LNK) $(I_LNK_FLAGS) $^ -o $@

all: render interactive

clean:
	rm $(OBJ_DIR)/*.$(OBJ_EXT)
	rm $(BIN)
	rm imgs/* movs/*
