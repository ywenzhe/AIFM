CXX = g++
# It's okay to have -pthread here for compiler-specific thread macros/warnings
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -pthread

# If Crypto++ or Snappy headers/libs are in non-standard locations, add -I and -L flags:
# Example: CXXFLAGS += -I/path/to/cryptopp/include -I/path/to/snappy/include
# LDFLAGS += -L/path/to/cryptopp/lib -L/path/to/snappy/lib

# Add -pthread to the linker flags as well
LDFLAGS = -pthread
LIBS = -lsnappy -lcryptopp

TARGET = fm_benchmark

SRCS = main.cpp
OBJS = $(SRCS:.cpp=.o)

all: $(TARGET)

# The most important change is here, adding $(LDFLAGS) to the link command
$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $@ $(LDFLAGS) $(LIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
