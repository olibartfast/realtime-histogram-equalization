#include "HistogramCPU.hpp"
#include <stdexcept>
#include <numeric>

void HistogramCPU::equalize(cv::Mat& outputImage, const cv::Mat& inputImage) {
    if (inputImage.empty() || inputImage.type() != CV_8UC3) {
        throw std::invalid_argument("Input image must be a non-empty 8-bit, 3-channel image");
    }

    int width = inputImage.cols;
    int height = inputImage.rows;
    int total_pixels = width * height;

    std::vector<unsigned char> grayImage(total_pixels);
    std::vector<unsigned int> histogram(HISTOGRAM_LENGTH, 0);
    std::vector<float> cdf(HISTOGRAM_LENGTH);

    h_RGBtoGS(grayImage, inputImage);
    h_Histogram(histogram, grayImage);
    h_Cdf(cdf, histogram, width, height);
    float cdfmin = h_MinOfCdf(cdf);

    outputImage = inputImage.clone();
    h_Hef(outputImage, cdf, cdfmin);
}

float HistogramCPU::h_p(unsigned int x, int width, int height) {
    return static_cast<float>(x) / static_cast<float>(width * height);
}

float HistogramCPU::h_clamp(float x, float start, float end) {
    return std::min(std::max(x, start), end);
}

void HistogramCPU::h_RGBtoGS(std::vector<unsigned char>& grayImage, const cv::Mat& inputImage) const {
    int width = inputImage.cols;
    int height = inputImage.rows;
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            const cv::Vec3b& pixel = inputImage.at<cv::Vec3b>(i, j);
            grayImage[i * width + j] = static_cast<unsigned char>(
                0.21f * pixel[2] + 0.71f * pixel[1] + 0.07f * pixel[0]
            );
        }
    }
}

void HistogramCPU::h_Histogram(std::vector<unsigned int>& histogram, const std::vector<unsigned char>& grayImage) const {
    for (unsigned char pixel : grayImage) {
        ++histogram[pixel];
    }
}

void HistogramCPU::h_Cdf(std::vector<float>& cdf, const std::vector<unsigned int>& histogram, int width, int height) const {
    cdf[0] = h_p(histogram[0], width, height);
    for (size_t i = 1; i < HISTOGRAM_LENGTH; ++i) {
        cdf[i] = cdf[i-1] + h_p(histogram[i], width, height);
    }
}

float HistogramCPU::h_MinOfCdf(const std::vector<float>& cdf) const {
    return *std::min_element(cdf.begin(), cdf.end());
}

void HistogramCPU::h_Hef(cv::Mat& outputImage, const std::vector<float>& cdf, float cdfmin) const {
    int width = outputImage.cols;
    int height = outputImage.rows;
    int channels = outputImage.channels();

    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            int z = (j * width + i) * channels;
            for (int c = 0; c < channels; ++c) {
                outputImage.data[z + c] = static_cast<unsigned char>(
                    h_clamp(255.0f * (cdf[outputImage.data[z + c]] - cdfmin) / (1.0f - cdfmin), 0.0f, 255.0f)
                );
            }
        }
    }
}