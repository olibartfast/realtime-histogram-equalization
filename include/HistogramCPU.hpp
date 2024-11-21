#pragma once

#include "Histogram.hpp"
#include <vector>
#include <cstddef>
#include <algorithm>
#include <cmath>

class HistogramCPU final : public Histogram {
public:
    HistogramCPU() = default;
    ~HistogramCPU() override = default;

    void equalize(cv::Mat& outputImage, const cv::Mat& inputImage) override;

private:
    static constexpr size_t HISTOGRAM_LENGTH = 256;
    static constexpr size_t BSIZE = 1024;

    static float h_p(unsigned int x, int width, int height);
    static float h_clamp(float x, float start, float end);
    void h_RGBtoGS(std::vector<unsigned char>& grayImage, const cv::Mat& inputImage) const;
    void h_Histogram(std::vector<unsigned int>& histogram, const std::vector<unsigned char>& grayImage) const;
    void h_Cdf(std::vector<float>& cdf, const std::vector<unsigned int>& histogram, int width, int height) const;
    float h_MinOfCdf(const std::vector<float>& cdf) const;
    void h_Hef(cv::Mat& outputImage, const std::vector<float>& cdf, float cdfmin) const;
};