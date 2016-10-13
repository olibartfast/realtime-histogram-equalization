OPENCV_LIBS=-lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_videoio -lopencv_imgproc

CUDA_INCLUDEPATH=/opt/cuda/include

NVCC_OPTS=-O3 -arch=sm_30 -x cu -use_fast_math 

all:
	nvcc -o out  $(NVCC_OPTS) main.cpp -I $(CUDA_INCLUDEPATH) $(OPENCV_LIBS)

clean:
	rm  out
