CXX = g++
# Aggressive optimization flags for maximum performance (GCC compatible)
CXXFLAGS = -std=c++17 -O3 -march=native -mtune=native -Wall -Wextra \
           -fopenmp -ffast-math -flto -funroll-loops \
           -finline-functions -ftree-vectorize -fomit-frame-pointer \
           -DNDEBUG
LDFLAGS = -fopenmp -flto

# Only compile the essential C++ files used in the analysis pipeline
ESSENTIAL_SOURCES = cpputils/fhn.cpp cpputils/linard.cpp cpputils/weighted_construct.cpp \
                   cpputils/unweighted_construct.cpp cpputils/graph_metrics.cpp
CPP_TARGETS = $(patsubst cpputils/%.cpp,%,$(ESSENTIAL_SOURCES))
TARGETS = $(CPP_TARGETS)

# Default target - build all executables
all: $(TARGETS)

# Rule to compile each .cpp file from cpputils to executable in root directory
%: cpputils/%.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

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