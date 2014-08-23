PGF90 = pgf90 -g -Mcuda=6.0,cc30,keep,rdc -tp=core2
NVCC = nvcc
GPUARCH = 30
BITS = 32
LDBITS =
ifeq ($(BITS), 32)
	LDBITS := -melf_i386
endif

.SUFFIXES: .cu .cuf .CUF

all: test_gpu test_cpu

#
# GPU test application
#

test_gpu: kernel1.gpu.o kernel2.gpu.o main_gpu.o asyncio.CUF.gpu.o asyncio.cu.gpu.o hooks.o
	$(PGF90) $^ -o $@ ~/forge/pgiwrapper/x86/libpgiwrapper.a -lgfortran -lgcc_s -lstdc++ -ldl -lelf -L$(shell dirname $(shell which nvcc))/../lib -lcudart_static -Wl,--wrap=__cudaRegisterVar

main_gpu.o: main_gpu.cuf
	$(PGF90) -g -m$(BITS) -c $< -o $@

kernel1.gpu.o: kernel1.CUF asyncio.CUF.gpu.o
	pgf90 -g -c $< -Mcuda=keep,cc30,rdc -o $<.o -DGPU && sed -i "s/_function_gpu_16/__FUNC_$(basename $<)/" $(basename $<).n001.gpu && ln -sf $(basename $<).n001.gpu $(basename $<).n001.cu && $(NVCC) -m$(BITS) -arch=sm_$(GPUARCH) --device-c -c $(basename $<).n001.cu -DPGI_CUDA_NO_TEXTURES -DCUDA_DOUBLE_MATH_FUNCTIONS -D__PGI_M32 -DGPU -o $@.int.o && objcopy --weaken $<.o && ld $(LDBITS) -r $@.int.o $<.o -o $@

kernel2.gpu.o: kernel2.CUF asyncio.CUF.gpu.o
	pgf90 -g -c $< -Mcuda=keep,cc30,rdc -o $<.o -DGPU && sed -i "s/_function_gpu_16/__FUNC_$(basename $<)/" $(basename $<).n001.gpu && ln -sf $(basename $<).n001.gpu $(basename $<).n001.cu && $(NVCC) -m$(BITS) -arch=sm_$(GPUARCH) --device-c -c $(basename $<).n001.cu -DPGI_CUDA_NO_TEXTURES -DCUDA_DOUBLE_MATH_FUNCTIONS -D__PGI_M32 -DGPU -o $@.int.o && objcopy --weaken $<.o && ld $(LDBITS) -r $@.int.o $<.o -o $@

asyncio.CUF.gpu.o: asyncio.CUF
	$(PGF90) -g -c $< -o $@ -DGPU

asyncio.cu.gpu.o: asyncio.cu
	$(NVCC) -DDYNAMIC -g -m$(BITS) -arch=sm_30 -rdc=true -c $< -o $@

#
# CPU test application
#

test_cpu: kernel1.cpu.o kernel2.cpu.o main_cpu.o asyncio.CUF.cpu.o asyncio.cu.cpu.o hooks.o
	$(PGF90) $^ -o $@ ~/forge/pgiwrapper/x86/libpgiwrapper.a -lgfortran -lgcc_s -lstdc++ -ldl -lelf

main_cpu.o: main_cpu.f90
	$(PGF90) -g -m$(BITS) -c $< -o $@

kernel1.cpu.o: kernel1.CUF asyncio.CUF.cpu.o
	$(PGF90) -g -m$(BITS) -c $< -o $@.int.o && objcopy --redefine-sym function_=$< --redefine-sym ..Dm_function=..Dm_$< $@.int.o $@

kernel2.cpu.o: kernel2.CUF asyncio.CUF.cpu.o
	$(PGF90) -g -m$(BITS) -c $< -o $@.int.o && objcopy --redefine-sym function_=$< --redefine-sym ..Dm_function=..Dm_$< $@.int.o $@

asyncio.CUF.cpu.o: asyncio.CUF
	$(PGF90) -g -m$(BITS) -c $< -o $@

asyncio.cu.cpu.o: asyncio.cu
	gcc -DDYNAMIC -g -m$(BITS) -x c++ -c $< -o $@

hooks.o: hooks.f90
	gfortran -g -m$(BITS) -c $< -o $@

clean:
	rm -rf test_gpu test_cpu *.o *.mod *.n001.*

