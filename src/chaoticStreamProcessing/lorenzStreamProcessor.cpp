#include "lorenzStreamProcessor.h"
#include <cstring>
#include <cmath>

lorenzStreamProcessor::lorenzStreamProcessor(int pixelCount) : m_size(pixelCount) {
    m_intStream = new uint32_t[m_size];
}

lorenzStreamProcessor::~lorenzStreamProcessor() {
    delete[] m_intStream;
}

void lorenzStreamProcessor::ingestRawStream(const double* rawLorenzStream) {
    for (int i = 0; i < m_size; ++i) {
        m_intStream[i] = extractIntegralValues(rawLorenzStream[i]);
    }
}

uint32_t lorenzStreamProcessor::extractIntegralValues(double floatingValue) {
    // 1. Isolate fractional coordinate
    double absVal = std::abs(floatingValue);
    double frac   = absVal - std::floor(absVal);

    // Scale by 2^48
    uint64_t scaled = static_cast<uint64_t>(frac * 281474976710656ULL);

    // Shift away the bottom 16 bits of RK4 truncation noise, leaving 32 pure bits
    return static_cast<uint32_t>((scaled >> 16) & 0xFFFFFFFF);
}