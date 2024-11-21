#pragma once
#include <opencv2/core.hpp>

class Histogram {
public:
    virtual ~Histogram() = default;
    virtual void equalize(cv::Mat& outputImage, const cv::Mat& inputImage) = 0;
};