cmake_minimum_required(VERSION 3.20)

project(realtime-histogram-equalization LANGUAGES CXX CUDA)

find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Set CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -use_fast_math")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G")

# Add include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Gather source files
file(GLOB_RECURSE CPP_SOURCES "src/*.cpp" "include/*.h")
file(GLOB_RECURSE CUDA_SOURCES "src/*.cu" "include/*.cuh")

message("CPP_SOURCES = ${CPP_SOURCES}")
message("CUDA_SOURCES = ${CUDA_SOURCES}")

# Create executable
add_executable(${PROJECT_NAME} ${CPP_SOURCES} ${CUDA_SOURCES})

# Enable debug info in device code for debug builds
target_compile_options(${PROJECT_NAME} PRIVATE "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>")
target_compile_options(${PROJECT_NAME} PRIVATE "$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-use_fast_math;-lineinfo>")

# Set CUDA properties for the target
set_target_properties(${PROJECT_NAME} PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES native)

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17 cuda_std_17)    

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE 
    ${OpenCV_LIBS}
    CUDA::cudart
)

# Set position-independent code for Unix/Apple
if(UNIX OR APPLE)
    set_property(TARGET ${PROJECT_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)
endif()