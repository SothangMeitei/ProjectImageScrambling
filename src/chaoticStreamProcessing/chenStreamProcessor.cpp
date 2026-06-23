#include "chenStreamProcessor.h"
#include <cstring>
#include <algorithm>

chenStreamProcessor::chenStreamProcessor(int streamSize) : m_size(streamSize) {
    m_flatMapping = new int[m_size];
    m_structArray = new mappingArrayValue[m_size];
}

chenStreamProcessor::~chenStreamProcessor() {
    delete[] m_flatMapping;
    delete[] m_structArray;
}

void chenStreamProcessor::ingestRawStream(const float* rawChenStream) {
    for (int i = 0; i < m_size; ++i) {
        m_structArray[i].previousIndex = i;
        
        // Fast Type-Punning: Float bits to sortable entropy
        uint32_t rawBits;
        std::memcpy(&rawBits, &rawChenStream[i], sizeof(float));
        // Invert sign bit so negative floats sort monotonically before positive floats
        rawBits ^= (rawBits >> 31) ? 0xFFFFFFFF : 0x80000000;
        m_structArray[i].mantissaChaos = rawBits; 
    }
}

void chenStreamProcessor::sortAndExtractMapping() {
    _radixSort();

    // Harvest the flat GPU permutation map
    for (int i = 0; i < m_size; ++i) {
        m_flatMapping[i] = m_structArray[i].previousIndex;
    }
}

void chenStreamProcessor::_radixSort() {
    // Safely allocate staging buffer on the heap
    mappingArrayValue* ping_pong_buffer = new mappingArrayValue[m_size];
    mappingArrayValue* input_array = m_structArray;

    // 4 passes of 8-bit Byte Radix Sort (O(N) stability)
    for (int k = 0; k < 4; ++k) {
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
            size_t dest_index = offsets[byte_digit]++; 
            ping_pong_buffer[dest_index] = input_array[i];
        }

        // Ping-pong pointers contiguously
        std::swap(input_array, ping_pong_buffer);
    }

    // Because 4 passes is even, input_array perfectly equals m_structArray here!
    delete[] ping_pong_buffer;
}