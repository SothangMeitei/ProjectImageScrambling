#include "lorenzStreamProcessor.h"
#include <cstring>

lorenzStreamProcessor::lorenzStreamProcessor(int pixelCount) : m_size(pixelCount) {
    m_byteStream = new unsigned char[m_size];
}

lorenzStreamProcessor::~lorenzStreamProcessor() {
    delete[] m_byteStream;
}

void lorenzStreamProcessor::ingestRawStream(const float* rawLorenzStream, int floatCount) {
    int byteIdx = 0;
    for (int i = 0; i < floatCount && byteIdx < m_size; ++i) {
        uint32_t bits;
        std::memcpy(&bits, &rawLorenzStream[i], sizeof(float));
        
        // Unpack raw chaotic mantissa bits contiguously into 8-bit VRAM slots
        m_byteStream[byteIdx++] = (bits      ) & 0xFF;
        if (byteIdx < m_size) m_byteStream[byteIdx++] = (bits >>  8) & 0xFF;
        if (byteIdx < m_size) m_byteStream[byteIdx++] = (bits >> 16) & 0xFF;
    }
}