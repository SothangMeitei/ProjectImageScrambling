#include"chenStreamProcessor.h"
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
        
        uint32_t rawBits;
        std::memcpy(&rawBits, &rawChenStream[i], sizeof(float));
        m_structArray[i].mantissaChaos = rawBits; 
    }
}

void chenStreamProcessor::sortAndExtractMapping() {
    //instead of this normal comparitive sorting we use the non comparative radix sort
    std::sort(m_structArray, m_structArray + m_size, [](const mappingArrayValue& a, const mappingArrayValue& b) {
        return a.mantissaChaos < b.mantissaChaos;
    });

    for (int i = 0; i < m_size; ++i) {
        m_flatMapping[i] = m_structArray[i].previousIndex;
    }
}

void chenStreamProcessor::_radixSort(int size , unsigned char* input_array){
    unsigned char* ping_pong_buffer;    //just need two buffers

    for(int k = 0 ; k < 4 ; ++k){
        // We are looking at Pass 'k' (where k is 0, 1, 2, or 3)
        int shift = k * 8; // Bit-shift offset: 0, 8, 16, or 24

        // 1. HISTOGRAM PASS: Count exactly how many times each byte (0-255) appears
        size_t counts[256] = {0};
        for (int i = 0; i < size; ++i) {
            unsigned char byte_digit = (input_array[i] >> shift) & 0xFF;
            counts[byte_digit]++;
        }

        // 2. PREFIX-SUM (The Genius Step): 
        // Convert the raw counts into the exact ending index where each bucket should sit in the output array.
        size_t offsets[256] = {0};
        offsets[0] = 0;
        for (int b = 1; b < 256; ++b) {
            offsets[b] = offsets[b - 1] + counts[b - 1];
        }

        // 3. ROUTING PASS: Drop the items directly into their calculated, stable offsets
        for (int i = 0; i < size; ++i) {
            unsigned char byte_digit = (input_array[i] >> shift) & 0xFF;
            
            // Grab the pre-calculated destination index, and instantly increment the offset for the next identical byte!
            size_t dest_index = offsets[byte_digit]++; 
            
            ping_pong_buffer[dest_index] = input_array[i];
        }
    }
}