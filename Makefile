CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall

# Find all .cpp and .cu files and create corresponding targets
CPP_SOURCES = $(wildcard *.cpp)
CPP_TARGETS = $(CPP_SOURCES:.cpp=)
TARGETS = $(CPP_TARGETS) 

# Default target - explicitly build all targets
all: $(TARGETS)

# Rule to compile each .cpp file to its corresponding executable
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%: %.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

# Clean target
clean:
	rm -f $(TARGETS)

# Create necessary directories
dirs:
	mkdir -p data plots data/fhn plots/fhn data/linard plots/linard

wipe:
	rm -f $(TARGETS)
	rm -rf data
	rm -rf plots

.PHONY: all clean dirs