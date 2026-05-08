#===============================================================================
# Makefile for LightGBM Fortran API
#===============================================================================

# Compiler settings
# Use Intel Fortran (ifx) or GNU Fortran (gfortran)
FC ?= ifx
#FC ?= gfortran

# Compiler flags
ifeq ($(FC),ifx)
    FFLAGS = -O2 -warn all -check bounds
    FFLAGS_DEBUG = -O0 -g -warn all -check all -traceback
else
    FFLAGS = -O2 -Wall -Wextra -fcheck=bounds
    FFLAGS_DEBUG = -O0 -g -Wall -Wextra -fcheck=all -fbacktrace
endif

# LightGBM paths (adjust these for your system)
LGBM_DIR ?= /usr/local
LGBM_INCLUDE = $(LGBM_DIR)/include
LGBM_LIB = $(LGBM_DIR)/lib

# Linker flags
LDFLAGS = -L$(LGBM_LIB) -llightgbm -Wl,-rpath,$(LGBM_LIB)

# Source files
API_SRC = lightgbm_api.f90
WRAPPER_SRC = lightgbm_wrapper.f90
EXAMPLE_SRC = lgbm_example.f90

# Object files
API_OBJ = lightgbm_api.o
WRAPPER_OBJ = lightgbm_wrapper.o
EXAMPLE_OBJ = lgbm_example.o

# Module files
MODS = lightgbm_api.mod lightgbm_wrapper.mod

# Targets
.PHONY: all clean debug example wrapper_example test help

all: example

# Build low-level API module
$(API_OBJ): $(API_SRC)
	$(FC) $(FFLAGS) -c $< -o $@

# Build high-level wrapper module
$(WRAPPER_OBJ): $(WRAPPER_SRC) $(API_OBJ)
	$(FC) $(FFLAGS) -c $< -o $@

# Build example program
$(EXAMPLE_OBJ): $(EXAMPLE_SRC) $(API_OBJ)
	$(FC) $(FFLAGS) -c $< -o $@

# Link example program
example: $(API_OBJ) $(EXAMPLE_OBJ)
	$(FC) $(FFLAGS) $^ -o lgbm_example $(LDFLAGS)
	@echo ""
	@echo "Build successful! Run with: ./lgbm_example"
	@echo "Make sure LightGBM library is in your library path."

# Build with wrapper module
wrapper_example: $(API_OBJ) $(WRAPPER_OBJ) lgbm_wrapper_example.o
	$(FC) $(FFLAGS) $^ -o lgbm_wrapper_example $(LDFLAGS)

lgbm_wrapper_example.o: lgbm_wrapper_example.f90 $(WRAPPER_OBJ)
	$(FC) $(FFLAGS) -c $< -o $@

# Debug build
debug: FFLAGS = $(FFLAGS_DEBUG)
debug: clean example

# Run tests
test: example
	@echo "Running example..."
	@LD_LIBRARY_PATH=$(LGBM_LIB):$$LD_LIBRARY_PATH ./lgbm_example

# Clean up
clean:
	rm -f *.o *.mod lgbm_example lgbm_wrapper_example lgbm_model.txt

# Help
help:
	@echo "LightGBM Fortran API Makefile"
	@echo ""
	@echo "Usage:"
	@echo "  make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  all              Build example program (default)"
	@echo "  example          Build example program"
	@echo "  wrapper_example  Build wrapper example program"
	@echo "  debug            Build with debug flags"
	@echo "  test             Build and run example"
	@echo "  clean            Remove build artifacts"
	@echo "  help             Show this help"
	@echo ""
	@echo "Variables:"
	@echo "  FC               Fortran compiler (default: ifx)"
	@echo "  LGBM_DIR         LightGBM installation directory (default: /usr/local)"
	@echo ""
	@echo "Examples:"
	@echo "  make FC=gfortran"
	@echo "  make LGBM_DIR=/opt/lightgbm"
	@echo "  make FC=ifx LGBM_DIR=~/lightgbm debug"
