# Compiler
CXX = g++
 
# C++ standard: C++17 for features like std::atomic_flag::test_and_set, etc.
# -O3 for optimization
# -Wall -Wextra for warnings
# -pthread for std::thread and other concurrency features
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -pthread
 
# Define -D (preprocessor definitions) if needed, e.g., for enabling debug asserts
# DEFS = -DDEBUG
 
# Source file
SRCS = main.cpp
 
# Output executable name
TARGET = benchmark_app
 
.PHONY: all clean
 
all: $(TARGET)
 
$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(DEFS) $^ -o $@
 
clean:
	rm -f $(TARGET)