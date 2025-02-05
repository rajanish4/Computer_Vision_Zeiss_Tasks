# Makefile

# Compiler
CXX = g++
CXXFLAGS = -std=c++11 -pthread -Wall -Wextra -pedantic

# Directories
INCLUDE_DIR = include
SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin

# Files
TARGET = $(BIN_DIR)/integral_image_multicore
SOURCES = $(wildcard $(SRC_DIR)/*.cpp)
OBJECTS = $(SOURCES:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)

# Create directories if they don't exist
$(shell mkdir -p $(OBJ_DIR))
$(shell mkdir -p $(BIN_DIR))

# Build Target
all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -I$(INCLUDE_DIR) -o $@ $^

# Compile source files to object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -I$(INCLUDE_DIR) -c $< -o $@

# Clean Target
clean:
	rm -rf $(OBJ_DIR)/*.o $(TARGET)

.PHONY: all clean
