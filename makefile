CUDA_DIR := /usr/local/cuda-10.0/
CUDA_ARCH := sm_75

CXX := clang++
LNK := $(CUDA_DIR)bin/nvcc

CXX_FLAGS := -x cuda --cuda-path=$(CUDA_DIR) --cuda-gpu-arch=$(CUDA_ARCH) -fPIC -std=c++11
LNK_FLAGS := -L$(CUDA_DIR)lib64/ -lcudart_static -lncurses

SRC_EXT := cpp
OBJ_DIR := obj
OBJ_EXT := o

SRC := $(wildcard *.$(SRC_EXT))
OBJ := $(patsubst %.$(SRC_EXT), $(OBJ_DIR)/%.$(OBJ_EXT), $(SRC))
BIN := mandel

$(BIN): $(OBJ)
	$(LNK) $(LNK_FLAGS) $^ -o $@

$(OBJ_DIR)/%.$(OBJ_EXT): %.$(SRC_EXT)
	$(CXX) $(CXX_FLAGS) -c $< -o $@

clean:
	rm $(OBJ_DIR)/*.$(OBJ_EXT)
	rm $(BIN)
	rm imgs/*
	rm movs/*
