cmake_minimum_required(VERSION 3.5.1)

project (realtime-histogram-equalization C CXX)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

link_libraries(${OpenCV_LIBS} ${CUDA_LIBS})
include_directories(${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})



if(CUDA_FOUND)
  set(CUDA_NVCC_FLAGS
  ${CUDA_NVCC_FLAGS};
  -O3 -use_fast_math 
)

  # add debugging to CUDA NVCC flags.  For NVidia's NSight tools.
set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} "-G")

else(CUDA_FOUND)
  message("CUDA is not installed on this system.")
endif()


if(UNIX OR APPLE)
  list(APPEND CUDA_NVCC_FLAGS "-Xcompiler;-fPIC;")
endif()

file( GLOB src src/main.cu src/histoEqCuda.h src/histoEqCpu.h )
cuda_add_executable(rtheq  ${src})
