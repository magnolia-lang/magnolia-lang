TARGET := $(shell basename `pwd`).bin

.PHONY: all clean setup

BIN_DIR := bin
CPP_SOURCE_DIR := cpp-src
CPP_SOURCES := $(shell find $(CPP_SOURCE_DIR) -name "*.cpp")

CUDA_SOURCE_DIR := cuda-src
CUDA_SOURCES := $(shell find $(CUDA_SOURCE_DIR) -name "*.cu")

CC := clang++
CCFLAGS := -std=c++20 -I$(CPP_SOURCE_DIR)

CUCC := nvcc
CUCCFLAGS := -std=c++17 -I$(CUDA_SOURCE_DIR)

MKDIR := @mkdir
RM := @rm

all: setup
	g++ $(CCFLAGS) $(CPP_SOURCES) -o $(BIN_DIR)/$(TARGET)
	@echo $(TARGET)
	@echo $(CPP_SOURCES)

cuda:
	$(CUCC) $(CUCCFLAGS) $(CUDA_SOURCES) -o $(BIN_DIR)/$(TARGET)
	@echo $(TARGET)
	@echo $(CUDA_SOURCES)

setup:
	$(MKDIR) -p $(BIN_DIR)

clean:
	$(RM) -r $(BIN_DIR)
