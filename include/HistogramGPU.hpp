#pragma once
#include "Histogram.hpp"
#include <cuda_runtime.h>
#include "histoEqCuda.cuh"

class HistogramGPU final : public Histogram {
public:
    HistogramGPU();
    ~HistogramGPU() override;
    void equalize(cv::Mat& outputImage, const cv::Mat& inputImage) override;

private:
    static constexpr int HISTOGRAM_LENGTH = 256;
    static constexpr int BSIZE = 1024;
    static constexpr int BSIZE2D = 32;
    
    unsigned char* d_ucharImage;
    unsigned char* d_grayImage;
    unsigned int* d_histogram;
    float* d_cdf;
    float* d_cdfmin;
    cudaStream_t stream;

    void allocateMemory(int width, int height, int channels);
    void freeMemory();

    // CUDA wrapper functions
    void launchRGBtoGS(dim3 dimG, dim3 dimB, unsigned char* d_grayImage, unsigned char* d_ucharImage, int imageHeight, int imageWidth);
    void launchHistogram(dim3 dimGridHisto, dim3 dimBlockHisto, unsigned int* d_histogram, unsigned char* d_grayImage, int imageWidth, int imageHeight);
    void launchCdf(int blocks, int threads, float* d_cdf, unsigned int* d_histogram, int imageWidth, int imageHeight, int HISTOGRAM_LENGTH);
    void launchMinOfCdf(int blocks, int threads, float* d_cdf, float* d_cdfmin, int HISTOGRAM_LENGTH);
    void launchHef(dim3 dimGrid, dim3 dimBlock, unsigned char* d_ucharImage, float* d_cdf, float cdfmin, int imageWidth, int imageHeight, int imageChannels);
};