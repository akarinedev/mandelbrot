CUDA_DIR := /usr/local/cuda-10.0/
CUDA_ARCH := sm_75

CX := clang++
CX_FLAGS := -x cuda --cuda-path=$(CUDA_DIR) --cuda-gpu-arch=$(CUDA_ARCH)

LK := clang
LK_FLAGS := -L$(CUDA_DIR)lib64/ -lcudart_static -ldl -lrt -pthread

SRC_EXT := cpp
OBJ_DIR := obj
OBJ_EXT := o

SRC := $(wildcard *.$(SRC_EXT))
OBJ := $(patsubst %.$(SRC_EXT), $(OBJ_DIR)/%.$(OBJ_EXT), $(SRC))
BIN := mandel

$(BIN): $(OBJ)
	$(LK) $(LK_FLAGS) $^ -o $@

$(OBJ_DIR)/%.$(OBJ_EXT): %.$(SRC_EXT)
	$(CX) $(CX_FLAGS) -c $< -o $@

clean:
	rm $(OBJ_DIR)/*.$(OBJ_EXT)
	rm $(BIN)
