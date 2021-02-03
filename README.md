## dependencies
tested on Ubuntu 16.04  
CUDA(tested 9.2) OpenCV(tested 3.3.1) 

## compilation
### with Makefile:  
make  
### with Cmake:  
mkdir build  
cd build  
cmake ..  
make  

### Demo on YouTube
https://www.youtube.com/watch?v=LHSSxSk55hQ  with the following experimental setup:  

### Hardware setup
* Notebook Intel Core i7-3630QM and CPU @ 3.40 GHz, 4 cores and 8 threads, Ram 6 GB
* GPU Nvidia GeForce GT 640M, 709 MHz, 384 CUDA cores, Ram 2 GB

### Software setup
* Operating System Linux Archlinux
* CUDA SDK 8.0
* host compiler GCC 5.1
* device compiler NVCC 8.0.44 (with parameters: -O3 for host and device optimizations, -sm_30 for compute capability 3.0 support, -use_fast_math for fast math functions support)

Demo on YouTube using Jetson TK1: https://www.youtube.com/watch?v=Vje8XCtam7A
