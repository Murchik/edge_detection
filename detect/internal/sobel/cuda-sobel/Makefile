NVCC := nvcc

CSTD ?= c++14

NVCCFLAGS += --std=$(CSTD) -Werror cross-execution-space-call
NVCCFLAGS += -arch=sm_35 -Wno-deprecated-gpu-targets
NVCCFLAGS += -D_FORCE_INLINES -Xcompiler -fPIC

all: libsobel.so

sobel.o: sobel.cu sobel.h
	$(NVCC) $(NVCCFLAGS) -c $< 

libsobel.so: sobel.o
	$(NVCC) -shared -o $@ $<

clean:
	$(RM) sobel.o libsobel.so