.PHONY: all clean

SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c
.ONESHELL:
.DELETE_ON_ERROR:
MAKEFLAGS += --warn-undefined-variables

# Targets that don't generate dependency files.
NODEPS = clean

DIRS = obj

define Rule_Auto_Dependencies_base =
    $(1:.o=.d): $(2) | $(DIRS)
	@echo 'Generating dependencies file:  $(2)'
	gcc $$(CFLAGS) -MT '$(1:.o=.d)' -MM -MG '$(2)' -MF '$(1:.o=.d)'
    ifeq (0, $(words $(findstring $(MAKECMDGOALS),$(NODEPS))))
        -include $(1:.o=.d)
    endif
    $(1): $(1:.o=.d)
endef

define Rule_Auto_Dependencies =
    $(eval $(call Rule_Auto_Dependencies_base,$(1),$(2)))
    $(1): $(2)
endef

CPATH = 

library = ../lib


CC = gcc
# CC = clang

CPP = g++
# CPP = clang++

ARCH = $(shell uname -m)

CFLAGS = -Wall -Wextra
# Tells the compiler to use pipes instead of temporary files (faster compilation, but uses more memory).
CFLAGS += -pipe
CFLAGS += -fopenmp
ifeq ($(ARCH), x86_64)
    CFLAGS += -mbmi
    CFLAGS += -mbmi2
    CFLAGS += -march=native
	CFLAGS += -mavx2
endif
# CFLAGS += -g3 -fno-omit-frame-pointer
# CFLAGS += -Og
# CFLAGS += -O0
# CFLAGS += -O2
CFLAGS += -O3

# CFLAGS += -ffast-math

# CFLAGS += -flto
# CFLAGS += -march=native

CFLAGS += -D'LEVEL1_DCACHE_LINESIZE=$(shell getconf LEVEL1_DCACHE_LINESIZE)'
CFLAGS += -D'LEVEL1_DCACHE_SIZE=$(shell getconf LEVEL1_DCACHE_SIZE)'
CFLAGS += -D'LEVEL2_CACHE_SIZE=$(shell getconf LEVEL2_CACHE_SIZE)'
CFLAGS += -D'LEVEL3_CACHE_SIZE=$(shell getconf LEVEL3_CACHE_SIZE)'

CFLAGS += -I'$(library)'
CFLAGS += -D'_GNU_SOURCE'


# DOUBLE := 0
DOUBLE := 1

CFLAGS += -D'DOUBLE=$(DOUBLE)'
CFLAGS += -D'INT_T=int32_t'

ifeq ($(DOUBLE), 1)
	CFLAGS += -D'ValueType=double'
else
	CFLAGS += -D'ValueType=float'
endif


LDFLAGS =
LDFLAGS += -lm


# LIB_SRC = read_mtx.cpp pthread_functions.c omp_functions.c string_util.c io.c random.c parallel_io.c hash.c array_metrics.c plot.c csr.c csc.c kmeans.c kmeans_char.c
LIB_SRC = pthread_functions.c omp_functions.c string_util.c io.c parallel_io.c hash.c random.c array_metrics.c plot.c csr_util.c csr.c csr_converter.c matrix_market.c dlmc_matrix.c hardware_topology.c

LIB_OBJ := $(LIB_SRC)
LIB_OBJ := $(patsubst %.c,obj/%.o,$(LIB_OBJ))
LIB_OBJ := $(patsubst %.cpp,obj/%.o,$(LIB_OBJ))


EXE = mat_feat.exe 


all: $(EXE) | $(DIRS)

mat_feat.exe: mat_feat.cpp $(LIB_OBJ)
	$(CPP) $(CFLAGS) $^ -o $@ $(LDFLAGS)


%.exe: obj/%.o $(LIB_OBJ)
	$(CPP) $(CFLAGS) $^ -o $@ $(LDFLAGS)


$(call Rule_Auto_Dependencies,obj/mat_feat.o,mat_feat.cpp)
	$(CPP) $(CFLAGS) -c $< -o $@

# $(call Rule_Auto_Dependencies,obj/read_mtx.o,read_mtx.cpp)
# 	$(CPP) $(CFLAGS) -c $< -o $@

$(call Rule_Auto_Dependencies,obj/pthread_functions.o,$(library)/pthread_functions.c)
	$(CC) $(CFLAGS) -c $< -o $@
$(call Rule_Auto_Dependencies,obj/omp_functions.o,$(library)/omp_functions.c)
	$(CC) $(CFLAGS) -c $< -o $@
$(call Rule_Auto_Dependencies,obj/string_util.o,$(library)/string_util.c)
	$(CC) $(CFLAGS) -c $< -o $@
$(call Rule_Auto_Dependencies,obj/io.o,$(library)/io.c)
	$(CC) $(CFLAGS) -c $< -o $@
$(call Rule_Auto_Dependencies,obj/parallel_io.o,$(library)/parallel_io.c)
	$(CC) $(CFLAGS) -c $< -o $@
$(call Rule_Auto_Dependencies,obj/hash.o,$(library)/hash/hash.c)
	$(CC) $(CFLAGS) -c $< -o $@
$(call Rule_Auto_Dependencies,obj/random.o,$(library)/random.c)
	$(CC) $(CFLAGS) -c $< -o $@
$(call Rule_Auto_Dependencies,obj/array_metrics.o,$(library)/array_metrics.c)
	$(CC) $(CFLAGS) -c $< -o $@
$(call Rule_Auto_Dependencies,obj/plot.o,$(library)/plot/plot.c)
	$(CC) $(CFLAGS) -c $< -o $@
$(call Rule_Auto_Dependencies,obj/csr.o,csr.c)
	$(CC) $(CFLAGS) -c $< -o $@
$(call Rule_Auto_Dependencies,obj/csc.o,csc.c)
	$(CC) $(CFLAGS) -c $< -o $@
$(call Rule_Auto_Dependencies,obj/kmeans.o,$(library)/kmeans/kmeans.c)
	$(CC) $(CFLAGS) -c $< -o $@
$(call Rule_Auto_Dependencies,obj/kmeans_char.o,$(library)/kmeans/kmeans_char.c)
	$(CC) $(CFLAGS) -c $< -o $@
$(call Rule_Auto_Dependencies,obj/csr_converter.o,$(library)/aux/csr_converter.c,$(CFLAGS))
	$(CC) $(CFLAGS) -c $< -o $@
$(call Rule_Auto_Dependencies,obj/csr_util.o,$(library)/aux/csr_util.c,$(CFLAGS))
	$(CC) $(CFLAGS) -c $< -o $@
# $(call Rule_Auto_Dependencies,obj/csr_util_gen.o,$(library)/storage_formats/csr_util/csr_util_gen.c,$(CFLAGS))
# 	$(CC) $(CFLAGS) -c $< -o $@
$(call Rule_Auto_Dependencies,obj/matrix_market.o,$(library)/storage_formats/matrix_market/matrix_market.c,$(CFLAGS))
	$(CC) $(CFLAGS) -c $< -o $@
$(call Rule_Auto_Dependencies,obj/dlmc_matrix.o,$(library)/storage_formats/dlmc_matrices/dlmc_matrix.c,$(CFLAGS))
	$(CC) $(CFLAGS) -c $< -o $@
$(call Rule_Auto_Dependencies,obj/hardware_topology.o,$(library)/topology/hardware_topology.c,$(CFLAGS))
	$(CC) $(CFLAGS) -c $< -o $@

$(DIRS): %:
	mkdir -p $@

clean:
	$(RM) obj/*.o obj/*.d *.o *.exe a.out