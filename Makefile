OPENCV_LIBS=-lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_videoio -lopencv_imgproc 
CUDA_INCLUDEPATH=/usr/local/cuda/include
CUDA_LIBPATH=/usr/local/cuda/lib64 
CUDA_LIBS = -lcudart -lcuda

NVCC_OPTS=-O3 -x cu -use_fast_math 

all:
	nvcc -o out  $(NVCC_OPTS) src/main.cpp -I $(CUDA_INCLUDEPATH) $(OPENCV_LIBS) $(CUDA_LIBS) -L$(CUDA_LIBPATH)

clean:
	rm  out
