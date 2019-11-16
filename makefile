CUDA_DIR := /usr/local/cuda-10.0/
CUDA_ARCH := sm_75

CXX := clang++-8
CXX_FLAGS := -x cuda --cuda-path=$(CUDA_DIR) --cuda-gpu-arch=$(CUDA_ARCH) -fPIC -std=c++11 -I libs/ -g
OPT_FLAGS := 

I_LNK := $(CUDA_DIR)bin/nvcc
I_LNK_FLAGS := -L$(CUDA_DIR)lib64/ -lcudart_static -lncurses -g

P_LNK := $(CUDA_DIR)bin/nvcc
P_LNK_FLAGS := -L$(CUDA_DIR)lib64/ -lcudart_static -g

R_LNK := $(CUDA_DIR)bin/nvcc
R_LNK_FLAGS := -L$(CUDA_DIR)lib64/ -lcudart_static -g


SRC_DIR := src
SRC_EXT := cpp
OBJ_DIR := obj
OBJ_EXT := o

SRC := $(wildcard $(SRC_DIR)/*.$(SRC_EXT))
OBJ := $(patsubst $(SRC_DIR)/%.$(SRC_EXT), $(OBJ_DIR)/%.$(OBJ_EXT), $(SRC))

#CPP to O
$(OBJ_DIR)/%.$(OBJ_EXT): $(SRC_DIR)/%.$(SRC_EXT)
	$(CXX) $(CXX_FLAGS) $(OPT_FLAGS) -c $< -o $@

all: interactive prepare render

interactive: obj/interactive.o obj/gpu.o
	$(I_LNK) $(I_LNK_FLAGS) $^ -o bin/interactive

prepare: obj/prepare.o
	$(R_LNK) $(R_LNK_FLAGS) $^ -o bin/prepare

render: obj/render.o obj/gpu.o
	$(R_LNK) $(R_LNK_FLAGS) $^ -o bin/render


clean:
	rm $(OBJ_DIR)/*.$(OBJ_EXT)
	rm bin/*
