CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall

# Find all .cpp files in cpputils directory and create corresponding targets
CPP_SOURCES = $(wildcard cpputils/*.cpp)
CPP_TARGETS = $(patsubst cpputils/%.cpp,%,$(CPP_SOURCES))
TARGETS = $(CPP_TARGETS)

# Default target - build all executables
all: $(TARGETS)

# Rule to compile each .cpp file from cpputils to executable in root directory
%: cpputils/%.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

# Clean target
clean:
	rm -f $(TARGETS)

# Create necessary directories
dirs:
	mkdir -p data plots data/fhn plots/fhn data/linard plots/linard

# Remove executables and directories
wipe:
	rm -f $(TARGETS)
	rm -rf data
	rm -rf plots

# Show what targets will be built
show-targets:
	@echo "CPP Sources: $(CPP_SOURCES)"
	@echo "Targets: $(TARGETS)"

.PHONY: all clean dirs wipe show-targets