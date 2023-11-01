# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

CXX = g++
CXXFLAGS = -fopenmp -O3 -Wall -shared -std=c++11 -fPIC
PYBIND11_INCLUDES = $(shell python3 -m pybind11 --includes)
PYTHON_EXTENSION_SUFFIX = $(shell python3-config --extension-suffix)
TARGET = cpp_shell_gen$(PYTHON_EXTENSION_SUFFIX)

all: $(TARGET)

$(TARGET): cpp_shell_gen.cpp
	$(CXX) $(CXXFLAGS) $(PYBIND11_INCLUDES) $< -o $@

clean:
	rm -f $(TARGET)
