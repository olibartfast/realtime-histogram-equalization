#include "HistogramFactory.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <cmath>

const float EPSILON = 3.0f;  // Tolerance threshold for floating-point comparisons

bool compareImages(const cv::Mat& img1, const cv::Mat& img2, float epsilon) {
    if (img1.size() != img2.size() || img1.type() != img2.type()) {
        return false;
    }

    cv::Mat diff;
    cv::absdiff(img1, img2, diff);
    cv::Scalar mean = cv::mean(diff);
    std::cout << "Mean difference: " << mean[0] << ", " << mean[1] << ", " << mean[2] << std::endl;
    return (mean[0] <= epsilon) && (mean[1] <= epsilon) && (mean[2] <= epsilon);
}

void runAndMeasure(const std::unique_ptr<Histogram>& equalizer, const cv::Mat& input, cv::Mat& output, const std::string& type) {
    auto start = std::chrono::high_resolution_clock::now();
    equalizer->equalize(output, input);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    std::cout << type << " implementation took " << elapsed.count() << " seconds" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " <input_image> <output_prefix>" << std::endl;
        return 1;
    }

    std::string inputPath = argv[1];
    std::string outputPrefix = argv[2];

    cv::Mat inputImage = cv::imread(inputPath, cv::IMREAD_COLOR);
    if (inputImage.empty()) {
        std::cerr << "Error: Could not read input image" << std::endl;
        return 1;
    }

    cv::Mat outputCPU(inputImage.rows, inputImage.cols, inputImage.type());
    cv::Mat outputGPU(inputImage.rows, inputImage.cols, inputImage.type());

    try {
        auto cpuEqualizer = HistogramFactory::createHistogram("CPU");
        auto gpuEqualizer = HistogramFactory::createHistogram("GPU");

        runAndMeasure(cpuEqualizer, inputImage, outputCPU, "CPU");
        runAndMeasure(gpuEqualizer, inputImage, outputGPU, "GPU");

        bool resultsEqual = compareImages(outputCPU, outputGPU, EPSILON);
        
        if (resultsEqual) {
            std::cout << "CPU and GPU results are equal within the tolerance of " << EPSILON << std::endl;
        } else {
            std::cout << "CPU and GPU results differ beyond the tolerance of " << EPSILON << std::endl;
        }

        cv::imwrite(outputPrefix + "_cpu.jpg", outputCPU);
        cv::imwrite(outputPrefix + "_gpu.jpg", outputGPU);
        std::cout << "Output images saved with prefix: " << outputPrefix << std::endl;

        // Optionally, save a difference image to visualize any discrepancies
        cv::Mat diffImage;
        cv::absdiff(outputCPU, outputGPU, diffImage);
        cv::imwrite(outputPrefix + "_diff.jpg", diffImage);
        std::cout << "Difference image saved as: " << outputPrefix << "_diff.jpg" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
