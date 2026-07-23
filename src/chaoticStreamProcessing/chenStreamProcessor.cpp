#include "chenStreamProcessor.h"
#include <cstring>
#include <algorithm>
#include <cmath>

chenStreamProcessor::chenStreamProcessor(int streamSize) : m_size(streamSize) {
    m_flatMapping = new int[m_size];
    m_structArray = new mappingArrayValue[m_size];
}

chenStreamProcessor::~chenStreamProcessor() {
    delete[] m_flatMapping;
    delete[] m_structArray;
}

void chenStreamProcessor::ingestRawStream(const double* rawChenStream) {
    for (int i = 0; i < m_size; ++i) {
        m_structArray[i].previousIndex = i;

        // 1. Isolate fractional coordinate (|x| - floor(|x|))
        double absVal = std::abs(rawChenStream[i]);
        double frac   = absVal - std::floor(absVal);

        // 2. Scale by 10^14 to project 14 decimal places into 64-bit integer space
        uint64_t scaledChaos = static_cast<uint64_t>(frac * 1e14);

        m_structArray[i].mantissaChaos = scaledChaos;
    }
}

void chenStreamProcessor::sortAndExtractMapping() {
    _radixSort();
    for (int i = 0; i < m_size; ++i) {
        m_flatMapping[i] = m_structArray[i].previousIndex;
    }
}

void chenStreamProcessor::_radixSort() {
    mappingArrayValue* ping_pong_buffer = new mappingArrayValue[m_size];
    mappingArrayValue* input_array      = m_structArray;

    // 8 passes of 8-bit Byte Radix Sort (Fully sorts 64-bit uint64_t keys)
    for (int k = 0; k < 8; ++k) {
        int shift = k * 8;
        size_t counts[256] = {0};

        for (int i = 0; i < m_size; ++i) {
            unsigned char byte_digit = (input_array[i].mantissaChaos >> shift) & 0xFF;
            counts[byte_digit]++;
        }

        size_t offsets[256] = {0};
        offsets[0] = 0;
        for (int b = 1; b < 256; ++b) {
            offsets[b] = offsets[b - 1] + counts[b - 1];
        }

        for (int i = 0; i < m_size; ++i) {
            unsigned char byte_digit = (input_array[i].mantissaChaos >> shift) & 0xFF;
            size_t dest_index       = offsets[byte_digit]++;
            ping_pong_buffer[dest_index] = input_array[i];
        }

        // Ping-pong buffer pointers
        std::swap(input_array, ping_pong_buffer);
    }

    // After 8 even passes, input_array matches m_structArray
    delete[] ping_pong_buffer;
}