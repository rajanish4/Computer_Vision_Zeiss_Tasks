# Makefile

# Define CUDA_PATH for Windows
#CUDA_PATH := "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0"

# NVCC Compiler
NVCC := $(CUDA_PATH)/bin/nvcc.exe

# Compiler Flags
NVCC_FLAGS := -O3 -arch=sm_75

# Include Directories
INCLUDES := -I./include

# Source Files
SOURCES := src/scan_kernels.cu src/transpose_kernels.cu src/utility.cu src/main.cu

# Output Directory
BIN_DIR := bin

# Executable Name (Stored in bin/)
EXECUTABLE := $(BIN_DIR)/integral_image.exe

# Ensure bin/ exists
$(shell mkdir -p $(BIN_DIR))

# Build Target
all: $(EXECUTABLE)

$(EXECUTABLE): $(SOURCES) | $(BIN_DIR)
	@echo "Compiling CUDA sources..."
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $^
	@echo "Compilation completed. Executable stored in $(BIN_DIR)/"

# Ensure bin directory exists
$(BIN_DIR):
	@mkdir -p $(BIN_DIR)

# Clean Target
clean:
	@echo "Cleaning up..."
	del /Q $(BIN_DIR)\*.exe
	@echo "Clean completed."

# Phony Targets
.PHONY: all clean
