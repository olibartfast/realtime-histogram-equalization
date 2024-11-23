#include "HistogramFactory.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <cmath>
#include <random>
#include <iomanip>

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

void processImage(const cv::Mat& input, const std::string& outputPrefix) 
{
    cv::Mat outputCPU(input.rows, input.cols, input.type());
    cv::Mat outputGPU(input.rows, input.cols, input.type());

    try {
        auto cpuEqualizer = HistogramFactory::createHistogram("CPU");
        auto gpuEqualizer = HistogramFactory::createHistogram("GPU");

        // Profile CPU implementation
        auto cpuStart = std::chrono::high_resolution_clock::now();
        cpuEqualizer->equalize(outputCPU, input);
        auto cpuEnd = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> cpuElapsed = cpuEnd - cpuStart;

        // Profile GPU implementation
        auto gpuStart = std::chrono::high_resolution_clock::now();
        gpuEqualizer->equalize(outputGPU, input);
        auto gpuEnd = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> gpuElapsed = gpuEnd - gpuStart;

        // Print timing results
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "CPU implementation took " << cpuElapsed.count() << " ms" << std::endl;
        std::cout << "GPU implementation took " << gpuElapsed.count() << " ms" << std::endl;
        
        double speedup = cpuElapsed.count() / gpuElapsed.count();
        std::cout << "GPU speedup: " << speedup << "x" << std::endl;

        bool resultsEqual = compareImages(outputCPU, outputGPU, EPSILON);
        
        if (resultsEqual) {
            std::cout << "CPU and GPU results are equal within the tolerance of " << EPSILON << std::endl;
        } else {
            std::cout << "CPU and GPU results differ beyond the tolerance of " << EPSILON << std::endl;
        }

        cv::imwrite(outputPrefix + "_cpu.jpg", outputCPU);
        cv::imwrite(outputPrefix + "_gpu.jpg", outputGPU);
        std::cout << "Output images saved with prefix: " << outputPrefix << std::endl;

        // Save a difference image to visualize any discrepancies
        cv::Mat diffImage;
        cv::absdiff(outputCPU, outputGPU, diffImage);
        cv::imwrite(outputPrefix + "_diff.jpg", diffImage);
        std::cout << "Difference image saved as: " << outputPrefix << "_diff.jpg" << std::endl;

        // Create a comparison image with labels and timing information
        cv::Mat comparisonImage;
        cv::hconcat(input, outputCPU, comparisonImage);
        cv::hconcat(comparisonImage, outputGPU, comparisonImage);

        cv::putText(comparisonImage, "Input", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
        cv::putText(comparisonImage, "CPU Output", cv::Point(input.cols + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
        cv::putText(comparisonImage, "GPU Output", cv::Point(2 * input.cols + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);

        std::stringstream ss;
        ss << "CPU: " << cpuElapsed.count() << " ms, GPU: " << gpuElapsed.count() << " ms, Speedup: " << speedup << "x";
        cv::putText(comparisonImage, ss.str(), cv::Point(10, input.rows - 10), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);

        cv::imwrite(outputPrefix + "_comparison.jpg", comparisonImage);
        std::cout << "Comparison image saved as: " << outputPrefix << "_comparison.jpg" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return;
    }
}

void processVideo(const std::string& inputPath, const std::string& outputPrefix) {
    cv::VideoCapture capture(inputPath);
    int frame_width = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = capture.get(cv::CAP_PROP_FPS);

    cv::VideoWriter outputVideo;
    outputVideo.open(outputPrefix + "_comparison.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(frame_width * 2, frame_height));

    auto gpuEqualizer = HistogramFactory::createHistogram("GPU");

    cv::Mat frame, outputGPU;
    int frameCount = 0;
    double totalTime = 0.0;

    while (true) {
        if (!capture.read(frame)) {
            break;
        }

        outputGPU = frame.clone();
        
        // Measure equalization time
        auto start = std::chrono::high_resolution_clock::now();
        gpuEqualizer->equalize(outputGPU, frame);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        
        totalTime += elapsed.count();
        frameCount++;

        cv::Mat comparisonFrame;    
        cv::hconcat(frame, outputGPU, comparisonFrame);

        // Add labels
        cv::putText(comparisonFrame, "Raw Input", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
        cv::putText(comparisonFrame, "Histogram Equalized", cv::Point(frame_width + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
        
        // Add inference time
        std::stringstream ss;
        ss << "Inference time: " << std::fixed << std::setprecision(2) << elapsed.count() << " ms";
        cv::putText(comparisonFrame, ss.str(), cv::Point(10, frame_height - 10), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);

        outputVideo.write(comparisonFrame);
    }

    outputVideo.release();
    capture.release();

    double avgTime = totalTime / frameCount;
    std::cout << "Comparison video saved: " << outputPrefix << "_comparison.mp4" << std::endl;
    std::cout << "Average inference time per frame: " << avgTime << " ms" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " <input_image> <output_prefix>" << std::endl;
        return 1;
    }

    std::string inputPath = argv[1];
    std::string outputPrefix = argv[2];

    // Check if input path is image or video
    if (inputPath.find(".jpg") != std::string::npos || inputPath.find(".png") != std::string::npos || inputPath.find(".jpeg") != std::string::npos) {

        cv::Mat inputImage = cv::imread(inputPath, cv::IMREAD_COLOR);
        if (inputImage.empty()) {
            std::cerr << "Error: Could not read input image" << std::endl;
            return 1;
        }

        processImage(inputImage, outputPrefix);

    }
    else 
    {
        processVideo(inputPath, outputPrefix); 
    }
    return 0;
}


