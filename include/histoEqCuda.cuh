#pragma once
#include <cuda_runtime.h>


__global__ void d_RGBtoGS(unsigned char* grayImage, unsigned char* ucharImage, int height, int width);
__global__ void d_Histogram(unsigned int *histogram, unsigned char *grayImage, int width, int height);
__global__ void d_Cdf(float* cdf, unsigned int* histogram, int width, int height, int bins);
__global__ void d_MinOfCdf(const float* const input, float * output, int len);
__global__ void d_Hef(unsigned char* ucharImage, float *cdf, float cdfmin, int width, int height, int channels);
