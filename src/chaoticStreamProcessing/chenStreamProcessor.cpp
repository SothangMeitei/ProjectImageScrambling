#include "chenStreamProcessor.h"
#include <cstring>
#include <algorithm>

inline uint64_t rotl64_Chen(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

inline uint64_t avalancheMix64_Chen(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

chenStreamProcessor::chenStreamProcessor(int streamSize) : m_size(streamSize) {
    m_flatMapping = new int[m_size];
    m_structArray = new mappingArrayValue[m_size];
    m_byteStream  = new unsigned char[m_size];
}

chenStreamProcessor::~chenStreamProcessor() {
    delete[] m_flatMapping;
    delete[] m_structArray;
    delete[] m_byteStream;
}

void chenStreamProcessor::ingestRawStream(const chaoticStreamChen<double>& stream) {
    // Golden ratio seed for the IIR Whitening Accumulator
    uint64_t feedbackAccumulator = 0x9E3779B97F4A7C15ULL; 

    for (int i = 0; i < m_size; ++i) {
        m_structArray[i].previousIndex = i;
        
        uint64_t bitsX, bitsY, bitsZ;
        std::memcpy(&bitsX, &stream.x[i], sizeof(double));
        std::memcpy(&bitsY, &stream.y[i], sizeof(double));
        std::memcpy(&bitsZ, &stream.z[i], sizeof(double));

        // 1. Bit-Rotated Phase-Space Coupling (ZERO MASKING - Utilizes 100% of IEEE-754 space!)
        uint64_t rawState = bitsX ^ rotl64_Chen(bitsY, 21) ^ rotl64_Chen(bitsZ, 42);

        // 2. Chaotic IIR Whitening: Fold current orbital state into rotating historical accumulator
        feedbackAccumulator = avalancheMix64_Chen(rawState ^ rotl64_Chen(feedbackAccumulator, 17));

        // 3. Populate permutation structures and extract decorrelated byte stream
        m_structArray[i].mantissaChaos = feedbackAccumulator;
        m_byteStream[i] = static_cast<unsigned char>((feedbackAccumulator >> 24) & 0xFF);
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
            offsets[b] = offsets[b - 1] + counts[b - 1];         }
        for (int i = 0; i < m_size; ++i) {
            unsigned char byte_digit = (input_array[i].mantissaChaos >> shift) & 0xFF;
            size_t dest_index       = offsets[byte_digit]++;
            ping_pong_buffer[dest_index] = input_array[i];
        }
        std::swap(input_array, ping_pong_buffer);
    }
    if (input_array != m_structArray) {
        std::memcpy(m_structArray, input_array, m_size * sizeof(mappingArrayValue));
        delete[] input_array;
    } else {
        delete[] ping_pong_buffer;
    }
}