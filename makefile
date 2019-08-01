CX := clang++
CX_FLAGS := -x cuda --cuda-path=/usr/local/cuda-10.0/ --cuda-gpu-arch=sm_75

LK := clang
LK_FLAGS := -L/usr/local/cuda10.0/lib64

SRC := $(wildcard *.cpp)
OBJ := $(patsubst %.cpp, obj/%.o, $(SRC))
BIN := mandel

$(BIN): $(OBJ)
	$(LK) $(LK_FLAGS) $^ -o $@

obj/%.o: %.cpp
	$(CX) $(CX_FLAGS) -c $< -o $@
