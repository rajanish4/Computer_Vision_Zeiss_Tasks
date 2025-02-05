# Makefile

# Compilers
CXX = g++
NVCC = nvcc

# Compiler Flags
CXXFLAGS = -std=c++11 -pthread -Wall -Wextra -pedantic
# NVCCFLAGS = -arch=sm_75 # -std=c++11 -O3 // for personal computer, adjust accordingly

# Directories
INCLUDE_DIR_SINGLE = cpu_implementation/single_core/include
INCLUDE_DIR_MULTI = cpu_implementation/multi_core/include
INCLUDE_DIR_CUDA = cuda_implementation/include

SRC_DIR_SINGLE = cpu_implementation/single_core/src
SRC_DIR_MULTI = cpu_implementation/multi_core/src
SRC_DIR_CUDA = cuda_implementation/src

OBJ_DIR = obj
BIN_DIR = bin

# Executable Names
EXEC_SINGLE = integral_image_singlecore
EXEC_MULTI = integral_image_multicore
EXEC_CUDA = integral_image_cuda

# Source Files
SRC_SINGLE = $(SRC_DIR_SINGLE)/single_main.cpp $(SRC_DIR_SINGLE)/integral_image.cpp
SRC_MULTI = $(SRC_DIR_MULTI)/multi_main.cpp $(SRC_DIR_MULTI)/prefix_sum.cpp
SRC_CUDA = $(SRC_DIR_CUDA)/main.cu $(SRC_DIR_CUDA)/scan_kernels.cu $(SRC_DIR_CUDA)/transpose_kernels.cu $(SRC_DIR_CUDA)/utility.cu

# Object Files
OBJ_SINGLE = $(OBJ_DIR)/single_main.o $(OBJ_DIR)/integral_image.o
OBJ_MULTI = $(OBJ_DIR)/multi_main.o $(OBJ_DIR)/prefix_sum.o
OBJ_CUDA = $(OBJ_DIR)/main.o $(OBJ_DIR)/scan_kernels.o $(OBJ_DIR)/transpose_kernels.o $(OBJ_DIR)/utility.o

# All Targets
all: $(BIN_DIR)/$(EXEC_SINGLE) $(BIN_DIR)/$(EXEC_MULTI) $(BIN_DIR)/$(EXEC_CUDA)

# Build Single-Core Executable
$(BIN_DIR)/$(EXEC_SINGLE): $(OBJ_SINGLE)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -I$(INCLUDE_DIR_SINGLE) -o $@ $^

# Build Multi-Core Executable
$(BIN_DIR)/$(EXEC_MULTI): $(OBJ_MULTI)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -I$(INCLUDE_DIR_MULTI) -o $@ $^

# Build CUDA Executable
$(BIN_DIR)/$(EXEC_CUDA): $(OBJ_CUDA)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -I$(INCLUDE_DIR_CUDA) -o $@ $^

# Compile Single-Core Source
$(OBJ_DIR)/single_main.o: $(SRC_DIR_SINGLE)/single_main.cpp
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -I$(INCLUDE_DIR_SINGLE) -c $< -o $@

$(OBJ_DIR)/integral_image.o: $(SRC_DIR_SINGLE)/integral_image.cpp
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -I$(INCLUDE_DIR_SINGLE) -c $< -o $@

# Compile Multi-Core Sources
$(OBJ_DIR)/multi_main.o: $(SRC_DIR_MULTI)/multi_main.cpp
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -I$(INCLUDE_DIR_MULTI) -c $< -o $@

$(OBJ_DIR)/prefix_sum.o: $(SRC_DIR_MULTI)/prefix_sum.cpp
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -I$(INCLUDE_DIR_MULTI) -c $< -o $@

# Compile CUDA Sources
$(OBJ_DIR)/main.o: $(SRC_DIR_CUDA)/main.cu
	@mkdir -p $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -I$(INCLUDE_DIR_CUDA) -c $< -o $@

$(OBJ_DIR)/scan_kernels.o: $(SRC_DIR_CUDA)/scan_kernels.cu
	@mkdir -p $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -I$(INCLUDE_DIR_CUDA) -c $< -o $@

$(OBJ_DIR)/transpose_kernels.o: $(SRC_DIR_CUDA)/transpose_kernels.cu
	@mkdir -p $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -I$(INCLUDE_DIR_CUDA) -c $< -o $@

$(OBJ_DIR)/utility.o: $(SRC_DIR_CUDA)/utility.cu
	@mkdir -p $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -I$(INCLUDE_DIR_CUDA) -c $< -o $@

# Clean Build Files
clean:
	rm -rf $(OBJ_DIR)/*.o $(BIN_DIR)/$(EXEC_SINGLE) $(BIN_DIR)/$(EXEC_MULTI) $(BIN_DIR)/$(EXEC_CUDA)

.PHONY: all clean
