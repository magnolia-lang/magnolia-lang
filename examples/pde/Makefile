TARGET := $(shell basename `pwd`).bin

.PHONY: all clean setup

BIN_DIR := bin
CPP_SOURCE_DIR := cpp-src
CPP_SOURCES := $(shell find $(CPP_SOURCE_DIR) -name "*.cpp")

CC := g++

# if clang++, can use
# -mllvm -inline-threshold=1000000000
CCFLAGS := -g -mtune=native -march=native -O3 -std=c++20 -I$(CPP_SOURCE_DIR) -fopenmp

MKDIR := @mkdir
RM := @rm

all: setup
	$(CC) $(CCFLAGS) $(CPP_SOURCES) -o $(BIN_DIR)/$(TARGET)
	@echo $(TARGET)
	@echo $(CPP_SOURCES)

setup:
	$(MKDIR) -p $(BIN_DIR)

clean:
	$(RM) -r $(BIN_DIR)
