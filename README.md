
# Histogram Equalization Comparison
This project compares CPU and GPU implementations of histogram equalization for images and videos.

## Features

- Image and video processing support
- CPU vs GPU histogram equalization comparison
- Execution time measurement
- Comparison outputs and difference images
- Side-by-side video processing

## Prerequisites

- OpenCV
- C++ compiler with C++11 support
- CUDA (for GPU implementation)

### Compile the project:
```
cmake ..   
cmake --build .
```

## Usage

### For Images

```
./histogram_equalizer <input_image_path> <output_prefix>
```

### For Videos

```
./histogram_equalizer <input_video_path> <output_prefix>
```

## Output

### Images

- `<output_prefix>_cpu.jpg`: CPU-processed image
- `<output_prefix>_gpu.jpg`: GPU-processed image
- `<output_prefix>_diff.jpg`: Difference between CPU and GPU results

### Videos

- `<output_prefix>_comparison.mp4`: Side-by-side raw vs equalized comparison video

