TARGET = libaspt_spmm.a

SRCS = $(wildcard *.cu) $(wildcard *.cc)
OBJS = $(patsubst %cu, %cu.o, $(SRCS)) $(patsubst %cc, %cc.o, $(SRCS))

CUDA_HOME=/usr/local/cuda
NVCC = $(CUDA_HOME)/bin/nvcc

NVCC_FLAGS = 
NVCC_FLAGS += -O3 -lcudart
NVCC_FLAGS += -shared -std=c++17
NVCC_FLAGS += -Xcompiler -fPIC 
NVCC_FLAGS += -Xcompiler -fopenmp
NVCC_FLAGS += -ccbin /usr/bin
NVCC_FLAGS += -m64
NVCC_FLAGS += -gencode arch=compute_80,code=sm_80
NVCC_FLAGS += -diag-suppress=191,177
NVCC_FLAGS += -DDOUBLE=$(DOUBLE)

CC_FLAGS = -O3 -Wall -std=c++17 -fPIC -DDOUBLE=$(DOUBLE)

INC = -I$(CUDA_HOME)/include

all: $(TARGET)

$(TARGET): $(OBJS)
	ar cr $@ $^
	ranlib $@

%.cu.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(INC) -c $< -o $@

%.cc.o: %.cc
	$(CPP) $(CC_FLAGS) $(INC) -c $< -o $@

clean:
	rm -f *.o $(TARGET)
