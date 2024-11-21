#pragma once

#include "Histogram.hpp"
#include <vector>

class HistogramCPU final : public Histogram {
public:
    HistogramCPU() = default;
    ~HistogramCPU() override = default;

    void equalize(cv::Mat& outputImage, const cv::Mat& inputImage) override;

private:
    float h_p(unsigned int x, int width, int height);
    float h_clamp(float x, float start, float end);
    void h_RGBtoGS(unsigned char* grayImage, unsigned char* ucharImage, int height, int width);
    void h_Histogram(unsigned int* histogram, unsigned char* grayImage, int width, int height);
    void h_Cdf(float* cdf, unsigned int* histogram, int width, int height);
    float h_MinOfCdf(float* cdf);
    void h_Hef(unsigned char* ucharImage, float* cdf, float cdfmin, int width, int height, int channels);
};