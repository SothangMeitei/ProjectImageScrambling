#include "lorenzStreamProcessor.h"
#include <cstring>

lorenzStreamProcessor::lorenzStreamProcessor(int pixelCount) : m_size(pixelCount) {
    m_intStream = new uint32_t[m_size];
}

lorenzStreamProcessor::~lorenzStreamProcessor() {
    delete[] m_intStream;
}

void lorenzStreamProcessor::ingestRawStream(const float* rawLorenzStream) {
    for(int i = 0; i < m_size; ++i) {
        m_intStream[i] = extractIntegralValues(rawLorenzStream[i]);
    }
}

uint32_t lorenzStreamProcessor::extractIntegralValues(float floatingValue) {
    uint32_t k;
    std::memcpy(&k, &floatingValue, sizeof(float));

    // Austin Appleby's fmix32 mantissa avalanche finalizer
    k ^= k >> 16;
    k *= 0x85ebca6b;
    k ^= k >> 13;
    k *= 0xc2b2ae35;
    k ^= k >> 16;

    return k; // Return 1 word containing 4 dense, un-zeroed chaotic bytes
}