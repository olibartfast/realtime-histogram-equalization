#include "HistogramCPU.hpp"
#include <algorithm>
#include <cstring>


void HistogramCPU::equalize(cv::Mat& outputImage, const cv::Mat& inputImage) {
    int width = inputImage.cols;
    int height = inputImage.rows;
    int channels = inputImage.channels();

    std::vector<unsigned char> grayImage(width * height);
    std::vector<unsigned int> histogram(256, 0);
    std::vector<float> cdf(256);

    // Convert to grayscale
    h_RGBtoGS(grayImage.data(), inputImage.data, height, width);

    // Compute histogram
    h_Histogram(histogram.data(), grayImage.data(), width, height);

    // Compute CDF
    h_Cdf(cdf.data(), histogram.data(), width, height);

    // Find minimum of CDF
    float cdfmin = h_MinOfCdf(cdf.data());

    // Apply histogram equalization
    h_Hef(outputImage.data, cdf.data(), cdfmin, width, height, channels);
}

float HistogramCPU::h_p(unsigned int x, int width, int height) {
    return static_cast<float>(x) / static_cast<float>(width * height);
}

float HistogramCPU::h_clamp(float x, float start, float end) {
    return std::min(std::max(x, start), end);
}

void HistogramCPU::h_RGBtoGS(unsigned char* grayImage, unsigned char* ucharImage, int height, int width) {
    for (int ii = 0; ii < height; ii++) {
        for (int jj = 0; jj < width; jj++) {
            int idx = ii * width + jj;
            unsigned char r = ucharImage[3 * idx];
            unsigned char g = ucharImage[3 * idx + 1];
            unsigned char b = ucharImage[3 * idx + 2];
            grayImage[idx] = static_cast<unsigned char>(0.21f * r + 0.71f * g + 0.07f * b);
        }
    }
}

void HistogramCPU::h_Histogram(unsigned int* histogram, unsigned char* grayImage, int width, int height) {
    std::memset(histogram, 0, 256 * sizeof(unsigned int));
    for (int ii = 0; ii < width * height; ii++) {
        histogram[grayImage[ii]]++;
    }
}

void HistogramCPU::h_Cdf(float* cdf, unsigned int* histogram, int width, int height) {
    cdf[0] = h_p(histogram[0], width, height);
    for (int ii = 1; ii < 256; ii++) {
        cdf[ii] = cdf[ii - 1] + h_p(histogram[ii], width, height);
    }
}

float HistogramCPU::h_MinOfCdf(float* cdf) {
    return *std::min_element(cdf, cdf + 256);
}

void HistogramCPU::h_Hef(unsigned char* ucharImage, float* cdf, float cdfmin, int width, int height, int channels) {
    for (int ii = 0; ii < (width * height * channels); ii++) {
        ucharImage[ii] = static_cast<unsigned char>(h_clamp(255.0f * (cdf[ucharImage[ii]] - cdfmin) / (1.0f - cdfmin), 0.0f, 255.0f));
    }
}