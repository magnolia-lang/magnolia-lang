TARGET := $(shell basename `pwd`).bin

.PHONY: all clean setup

BIN_DIR := bin
CPP_SOURCE_DIR := cpp-src
CPP_SOURCES := $(shell find $(CPP_SOURCE_DIR) -name "*.cpp")

CUDA_SOURCE_DIR := cuda-src
CUDA_SOURCES := $(shell find $(CUDA_SOURCE_DIR) -name "*.cu")

DEFINES := -DPAD0=0 -DPAD1=0 -DPAD2=0 -DPROGRAM_NAME=PDEProgramDNF

CC := g++
# if clang++, can use
# -mllvm -inline-threshold=1000000000
CCFLAGS := -g -mtune=native -march=native -O3 -std=c++20 -I$(CPP_SOURCE_DIR) -fopenmp

CUARCH := 61
CUCC := nvcc
CUCCFLAGS := -std=c++17 -I$(CUDA_SOURCE_DIR) -arch=sm_$(CUARCH) \
             -rdc=true -g -lcudadevrt -lineinfo -Xcompiler -fopenmp

MKDIR := @mkdir
RM := @rm

all: setup
	$(CC) $(CCFLAGS) $(DEFINES) $(CPP_SOURCES) -o $(BIN_DIR)/$(TARGET)
	@echo $(TARGET)
	@echo $(CPP_SOURCES)

cuda:
	$(CUCC) $(CUCCFLAGS) $(DEFINES) $(CUDA_SOURCES) -o $(BIN_DIR)/$(TARGET)
	@echo $(TARGET)
	@echo $(CUDA_SOURCES)

setup:
	$(MKDIR) -p $(BIN_DIR)

clean:
	$(RM) -r $(BIN_DIR)
