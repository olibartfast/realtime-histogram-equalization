// HistogramFactory.cpp
#include "HistogramFactory.hpp"
#include "HistogramCPU.hpp"
#include "HistogramGPU.hpp"

std::unique_ptr<Histogram> HistogramFactory::createHistogram(const std::string& type) {
    if (type == "CPU") {
        return std::make_unique<HistogramCPU>();
    } else if (type == "GPU") {
        return std::make_unique<HistogramGPU>();
    } else {
        throw std::invalid_argument("Invalid histogram type");
    }
}