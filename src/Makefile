# Prepare necessary flags for compilation
CUDA_HOME ?= /usr/local/cuda
CUDA_COMPUTE_CAPABILITY ?= sm_80

PYTHON_INC := $(shell python3 -c "import sysconfig; print(sysconfig.get_path('include'))")
PYBIND_INC := $(shell python3 -c "import pybind11; print(pybind11.get_include())")


NVCC=nvcc
NVCCFLAGS=-use_fast_math -std=c++17 --compiler-options "-fPIC" -O3 -arch=$(CUDA_COMPUTE_CAPABILITY)

CC=g++
PYFLAGS=-I. -I$(PYTHON_INC) -I$(PYBIND_INC) 
CUDA_RUNTIME=-I$(CUDA_HOME)/include/ -L$(CUDA_HOME)/lib64/ -lcudart

interpolate.o: interpolate.cu
	$(NVCC) $(NVCCFLAGS) -c interpolate.cu -o interpolate.o

voxelize.so: interpolate.o interpolate_py.cpp
	$(CC) -fPIC -shared -o voxelize.so interpolate_py.cpp interpolate.o $(PYFLAGS) $(CUDA_RUNTIME)


# Generate two points around 5,5,5 and set weights 0.5, 0.5
# Interpolate to 10*10*10
test: 
	rm -f voxelize.so
	make voxelize.so
	python -c """import numpy as np; \
  import voxelize; \
  coord = np.random.normal(size=(2,3), loc=5); \
	print('Processing coord: ', coord); \
	ret = voxelize.interpolate(coord, np.array([0.5, 0.5]), np.array([10, 10, 10]), spacing=1.0, cutoff=100, sigma=100); \
	print('The final sum of the out array: ', ret.astype(np.float32).sum()); \
	"""
	cp voxelize.so ../feater/voxelize.so

clean:
	rm -f voxelize.so *.o *.so

	

# hdf5: readdata.cpp
# 	${CC} -fPIC -shared -I/usr/include/hdf5/serial/ -o readdata.so readdata.cpp $(PYFLAGS)
# 	python3 -c "import readdata; readdata.readdata('/media/yzhang/MieT72/Data/feater_database_surf/TestSet_TYR.h5', 'vertices', 0, 1000)"
