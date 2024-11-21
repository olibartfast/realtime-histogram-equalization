// HistogramFactory.h
#pragma once
#include "Histogram.hpp"
#include <memory>
#include <string>

class HistogramFactory {
public:
    static std::unique_ptr<Histogram> createHistogram(const std::string& type);
};