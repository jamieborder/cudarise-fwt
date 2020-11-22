# makefile for the CUDA D Hello World test
# to run, execute: $LD_LIBRARY_PATH=$LD_LIBRARY_PATH:. ./hello

DMD ?= dmd
NVCC ?= nvcc
GPP ?= g++
GCC ?= gcc

NUMFWTS ?= 100
PA ?= 5
VERB ?= 1
NUMRUNS ?= 10

CUDA_LIB_DIR ?= /usr/local/cuda/lib64
CUDA_DIR := ../cuda_d
CUDA_FILES := $(CUDA_DIR)/cublas_api.d \
	$(CUDA_DIR)/cublasXt.d \
	$(CUDA_DIR)/cudalibxt.d \
	$(CUDA_DIR)/curand.d \
	$(CUDA_DIR)/cublas.d \
	$(CUDA_DIR)/cuComplex.d \
	$(CUDA_DIR)/cuda_profiler_api.d \
	$(CUDA_DIR)/vector_types.d \
	$(CUDA_DIR)/cublas_v2.d \
	$(CUDA_DIR)/cuda.d \
	$(CUDA_DIR)/cuda_runtime_api.d

OUTPUT = prog

.PHONY: clean install run

clean:
	- rm -f *.o
	- rm -f *.so
	- rm -f *~
	- rm -f $(OUTPUT)

install: launchFWT.d fwt.cu simdFWT.c
	@$(GCC) -mavx2 -c simdFWT.c
	@$(NVCC) --shared -o libhwkernel.so fwt.cu -w --compiler-options '-fPIC'  
	@$(DMD) -of$(OUTPUT) launchFWT.d simdFWT.o -I$(CUDA_DIR)/ $(CUDA_FILES) \
		-L-L$(CUDA_LIB_DIR) -L-lcuda -L-lcudart -L-L. -L-lhwkernel

# for some reason, phobos2 giving me grief on workstation
# -L-L$(CUDA_LIB_DIR) -L-lcuda -L-lcudart -L-lphobos2 -L-L. -L-lhwkernel

run: install
	@LD_LIBRARY_PATH=$LD_LIBRARY_PATH:. ./$(OUTPUT) $(NUMFWTS) $(PA) $(VERB) $(NUMRUNS)
