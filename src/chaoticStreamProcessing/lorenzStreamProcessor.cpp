#include "lorenzStreamProcessor.h"
#include <cstring>

inline uint64_t rotl64_Lorenz(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

inline uint64_t avalancheMix64_Lorenz(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

lorenzStreamProcessor::lorenzStreamProcessor(int pixelCount) : m_size(pixelCount) {
    m_intStream = new uint32_t[m_size];
}

lorenzStreamProcessor::~lorenzStreamProcessor() {
    delete[] m_intStream;
}

void lorenzStreamProcessor::ingestRawStream(const chaoticStreamLorenz<double>& stream) {
    // Diversified golden ratio seed for 4D Hyper-Lorenz IIR Accumulator
    uint64_t feedbackAccumulator = 0xD1B54A32D192ED03ULL; 

    for (int i = 0; i < m_size; ++i) {
        uint64_t bitsX, bitsY, bitsZ, bitsW;
        std::memcpy(&bitsX, &stream.x[i], sizeof(double));
        std::memcpy(&bitsY, &stream.y[i], sizeof(double));
        std::memcpy(&bitsZ, &stream.z[i], sizeof(double));
        std::memcpy(&bitsW, &stream.w[i], sizeof(double));

        // 1. 4-Way Bit-Rotated Phase-Space Coupling (ZERO MASKING!)
        uint64_t rawState = bitsX ^ rotl64_Lorenz(bitsY, 16) ^ rotl64_Lorenz(bitsZ, 32) ^ rotl64_Lorenz(bitsW, 48);

        // 2. Chaotic Digital Whitening Filter (IIR State-Feedback Chaining)
        feedbackAccumulator = avalancheMix64_Lorenz(rawState ^ rotl64_Lorenz(feedbackAccumulator, 19));

        // 3. Extract a dense, decorrelated 32-bit diffusion word
        m_intStream[i] = static_cast<uint32_t>((feedbackAccumulator >> 16) & 0xFFFFFFFF);
    }
}