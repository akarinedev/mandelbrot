CX := clang++
CX_FLAGS := -x cuda --cuda-path=/usr/local/cuda-10.0/ --cuda-gpu-arch=sm_75

LK := clang
LK_FLAGS := -L/usr/local/cuda10.0/lib64

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
