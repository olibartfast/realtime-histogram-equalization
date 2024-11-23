#include "HistogramFactory.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <cmath>
#include <random>

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


void processImage(const cv::Mat& input,  const std::string& outputPrefix) 
{
   
    cv::Mat outputCPU(input.rows, input.cols, input.type());
    cv::Mat outputGPU(input.rows, input.cols, input.type());

    try {
        auto cpuEqualizer = HistogramFactory::createHistogram("CPU");
        auto gpuEqualizer = HistogramFactory::createHistogram("GPU");

        runAndMeasure(cpuEqualizer, input, outputCPU, "CPU");
        runAndMeasure(gpuEqualizer, input, outputGPU, "GPU");

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


    capture.read(frame);
    outputGPU = frame.clone();

    capture.set(cv::CAP_PROP_POS_FRAMES, 0); // Reset to beginning for video processing

    while (true) {
        if (!capture.read(frame)) {
            break;
        }

        outputGPU = frame.clone();
        gpuEqualizer->equalize(outputGPU, frame);
        cv::Mat comparisonFrame;    
        cv::hconcat(frame, outputGPU, comparisonFrame); 
        outputVideo.write(comparisonFrame);

    }

    outputVideo.release();
    capture.release();
    std::cout << "Comparison video saved: " << outputPrefix << "_comparison.mp4" << std::endl;
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


