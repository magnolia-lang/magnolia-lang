TARGET := $(shell basename `pwd`).bin

.PHONY: all clean setup

BIN_DIR := bin
CPP_SOURCE_DIR := cpp-src
CPP_SOURCES := $(shell find $(CPP_SOURCE_DIR) -name "*.cpp")
DEFINES := -DBENCHMARK=0 -DBENCH_ALL=0 -DBFS=0 -DDFS=0 -DDIJKSTRA=0 \
           -DBELLMANFORD=0 -DPRIM=0 -DJOHNSON=0 \
	   -DNB_TEST_VERTICES=100000 -DNB_TEST_EDGES=1000000

CC := g++

# if clang++, can use
# -mllvm -inline-threshold=1000000000
CCFLAGS := -g -O3 -std=c++20 -I$(CPP_SOURCE_DIR) -fopenmp

MKDIR := @mkdir
RM := @rm

all: setup
	$(CC) $(CCFLAGS) $(DEFINES) $(CPP_SOURCES) -o $(BIN_DIR)/$(TARGET)
	@echo $(TARGET)
	@echo $(CPP_SOURCES)

setup:
	$(MKDIR) -p $(BIN_DIR)

clean:
	$(RM) -r $(BIN_DIR)
