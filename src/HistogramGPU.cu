#include "HistogramGPU.hpp"
#include <cstdio>

#define CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) \
    { \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(-10*error); \
    }\
}


HistogramGPU::HistogramGPU() : d_ucharImage(nullptr), d_grayImage(nullptr), d_histogram(nullptr), d_cdf(nullptr), d_cdfmin(nullptr) {
    CHECK(cudaStreamCreate(&stream));
}

HistogramGPU::~HistogramGPU() {
    freeMemory();
    CHECK(cudaStreamDestroy(stream));
}

void HistogramGPU::allocateMemory(int width, int height, int channels) {
    if (d_ucharImage == nullptr) {
        CHECK(cudaMalloc((void**)&d_ucharImage, sizeof(unsigned char) * width * height * channels));
        CHECK(cudaMalloc((void**)&d_grayImage, sizeof(unsigned char) * width * height));
        CHECK(cudaMalloc((void**)&d_histogram, sizeof(unsigned int) * HISTOGRAM_LENGTH));
        CHECK(cudaMalloc((void**)&d_cdf, sizeof(float) * HISTOGRAM_LENGTH));
        CHECK(cudaMalloc((void**)&d_cdfmin, sizeof(float)));
    }
}

void HistogramGPU::freeMemory() {
    if (d_ucharImage != nullptr) {
        CHECK(cudaFree(d_ucharImage));
        CHECK(cudaFree(d_grayImage));
        CHECK(cudaFree(d_histogram));
        CHECK(cudaFree(d_cdf));
        CHECK(cudaFree(d_cdfmin));
        d_ucharImage = nullptr;
        d_grayImage = nullptr;
        d_histogram = nullptr;
        d_cdf = nullptr;
        d_cdfmin = nullptr;
    }
}

void HistogramGPU::equalize(cv::Mat& outputImage, const cv::Mat& inputImage) {
    int imageWidth = inputImage.cols;
    int imageHeight = inputImage.rows;
    int imageChannels = inputImage.channels();

    allocateMemory(imageWidth, imageHeight, imageChannels);

    CHECK(cudaMemcpyAsync(d_ucharImage, inputImage.data, sizeof(unsigned char) * imageWidth * imageHeight * imageChannels, cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemsetAsync(d_histogram, 0, sizeof(unsigned int) * HISTOGRAM_LENGTH, stream));

    dim3 dimB(BSIZE2D, BSIZE2D);
    dim3 dimG((imageWidth + BSIZE2D - 1) / BSIZE2D, (imageHeight + BSIZE2D - 1) / BSIZE2D);
    launchRGBtoGS(dimG, dimB, d_grayImage, d_ucharImage, imageHeight, imageWidth);

    dim3 dimBlockHisto(BSIZE, 1, 1);
    dim3 dimGridHisto((imageWidth * imageHeight + BSIZE - 1) / BSIZE, 1, 1);
    launchHistogram(dimGridHisto, dimBlockHisto, d_histogram, d_grayImage, imageWidth, imageHeight);

    launchCdf(1, BSIZE, d_cdf, d_histogram, imageWidth, imageHeight, HISTOGRAM_LENGTH);

    launchMinOfCdf(1, BSIZE, d_cdf, d_cdfmin, HISTOGRAM_LENGTH);

    float cdfmin;
    CHECK(cudaMemcpyAsync(&cdfmin, d_cdfmin, sizeof(float), cudaMemcpyDeviceToHost, stream));

    dim3 dimBlock(BSIZE2D, BSIZE2D, 1);
    dim3 dimGrid((imageWidth + BSIZE2D - 1) / BSIZE2D, (imageHeight + BSIZE2D - 1) / BSIZE2D, 1);
    launchHef(dimGrid, dimBlock, d_ucharImage, d_cdf, cdfmin, imageWidth, imageHeight, imageChannels);

    CHECK(cudaMemcpyAsync(outputImage.data, d_ucharImage, sizeof(unsigned char) * imageWidth * imageHeight * imageChannels, cudaMemcpyDeviceToHost, stream));
    CHECK(cudaStreamSynchronize(stream));
}

// CUDA wrapper function implementations
void HistogramGPU::launchRGBtoGS(dim3 dimG, dim3 dimB, unsigned char* d_grayImage, unsigned char* d_ucharImage, int imageHeight, int imageWidth) {
    d_RGBtoGS<<<dimG, dimB, 0, stream>>>(d_grayImage, d_ucharImage, imageHeight, imageWidth);
    CHECK(cudaGetLastError());
}

void HistogramGPU::launchHistogram(dim3 dimGridHisto, dim3 dimBlockHisto, unsigned int* d_histogram, unsigned char* d_grayImage, int imageWidth, int imageHeight) {
    d_Histogram<<<dimGridHisto, dimBlockHisto, 0, stream>>>(d_histogram, d_grayImage, imageWidth, imageHeight);
    CHECK(cudaGetLastError());
}

void HistogramGPU::launchCdf(int blocks, int threads, float* d_cdf, unsigned int* d_histogram, int imageWidth, int imageHeight, int HISTOGRAM_LENGTH) {
    d_Cdf<<<blocks, threads, 0, stream>>>(d_cdf, d_histogram, imageWidth, imageHeight, HISTOGRAM_LENGTH);
    CHECK(cudaGetLastError());
}

void HistogramGPU::launchMinOfCdf(int blocks, int threads, float* d_cdf, float* d_cdfmin, int HISTOGRAM_LENGTH) {
    d_MinOfCdf<<<blocks, threads, 0, stream>>>(d_cdf, d_cdfmin, HISTOGRAM_LENGTH);
    CHECK(cudaGetLastError());
}

void HistogramGPU::launchHef(dim3 dimGrid, dim3 dimBlock, unsigned char* d_ucharImage, float* d_cdf, float cdfmin, int imageWidth, int imageHeight, int imageChannels) {
    d_Hef<<<dimGrid, dimBlock, 0, stream>>>(d_ucharImage, d_cdf, cdfmin, imageWidth, imageHeight, imageChannels);
    CHECK(cudaGetLastError());
}