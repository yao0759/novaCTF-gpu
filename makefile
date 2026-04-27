IDIR=${CURDIR}/src
TESTDIR=${CURDIR}/tests

program=novaCTF
USE_CUDA ?= 0

CPP_SOURCES := $(shell find $(IDIR) -name '*.cpp')
CUDA_SOURCES :=
CUDA_OBJECTS :=
OBJECTS := $(CPP_SOURCES:.cpp=.o)

CXX = g++ -O3 -s -DNDEBUG
CXXFLAGS = -std=c++14
NVCC ?= nvcc
NVCCFLAGS ?= -O3 -std=c++14 -DNDEBUG
LDFLAGS =
LIBS = -lfftw3 -lfftw3f

VALIDATION_PROGRAM = fft_backend_validation
VALIDATION_OBJECTS = $(TESTDIR)/compare_fft_backends.o $(IDIR)/fftRoutines.o

ifeq ($(OS),Windows_NT)
CUDA_PATH ?= $(CUDA_HOME)
CUDA_INCLUDE_DIR ?= $(CUDA_PATH)/include
CUDA_LIB_DIR ?= $(CUDA_PATH)/lib/x64
else
CUDA_PATH ?= $(if $(CUDA_HOME),$(CUDA_HOME),/usr/local/cuda)
CUDA_INCLUDE_DIR ?= $(CUDA_PATH)/include
CUDA_LIB_DIR ?= $(CUDA_PATH)/lib64
endif

ifeq ($(USE_CUDA),1)
CUDA_SOURCES += $(IDIR)/cudaFftRoutines.cu
CUDA_OBJECTS := $(CUDA_SOURCES:.cu=.o)
OBJECTS += $(CUDA_OBJECTS)
VALIDATION_OBJECTS += $(CUDA_OBJECTS)
CXXFLAGS += -DUSE_CUDA -I$(CUDA_INCLUDE_DIR)
NVCCFLAGS += -DUSE_CUDA -I$(IDIR) -I$(CUDA_INCLUDE_DIR)
LDFLAGS += -L$(CUDA_LIB_DIR) -Wl,-rpath,$(CUDA_LIB_DIR)
LIBS += -lcufft -lcudart
endif

all: ${program}

build: ${program}

debug: CXX = g++ -g -W -Wall -Werror
debug: NVCCFLAGS = -g -std=c++14 -DUSE_CUDA -I$(IDIR) -I$(CUDA_INCLUDE_DIR)
debug: LDFLAGS += -g
debug: ${program}

${program}: CXXFLAGS += $(foreach d, $(includepath), -I$d)
${program}: LDFLAGS += $(foreach d, $(libpath), -L$d)
${program}: $(OBJECTS)
	$(CXX) -o $@ $^ $(LDFLAGS) $(LIBS)

$(VALIDATION_PROGRAM): $(VALIDATION_OBJECTS)
	$(CXX) -o $@ $^ $(LDFLAGS) $(LIBS)

validate_gpu: $(VALIDATION_PROGRAM)
	./$(VALIDATION_PROGRAM)

clean:
	rm -f src/*.o tests/*.o ${program} ${VALIDATION_PROGRAM} src/*.d

$(IDIR)/%.o: $(IDIR)/%.cpp
	$(CXX) -MD -MP $(CXXFLAGS) -o $@ -c $< $(LDFLAGS)

$(TESTDIR)/%.o: $(TESTDIR)/%.cpp
	$(CXX) -MD -MP $(CXXFLAGS) -o $@ -c $< $(LDFLAGS)

$(IDIR)/%.o: $(IDIR)/%.cu
	$(NVCC) $(NVCCFLAGS) -o $@ -c $<
