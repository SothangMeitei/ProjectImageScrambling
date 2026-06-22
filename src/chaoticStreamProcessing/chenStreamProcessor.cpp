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
    _radixSort();

    for (int i = 0; i < m_size; ++i) {
        m_flatMapping[i] = m_structArray[i].previousIndex;
    }
}

void chenStreamProcessor::_radixSort(){
    mappingArrayValue* ping_pong_buffer;    //just need two buffers
    mappingArrayValue* input_array;

    for(int k = 0 ; k < 4 ; ++k){
        if(k & 2 == 1){
            auto temp{input_array};
            input_array = ping_pong_buffer;
            ping_pong_buffer = temp;
        }
        // We are looking at Pass 'k' (where k is 0, 1, 2, or 3)
        int shift = k * 8; // Bit-shift offset: 0, 8, 16, or 24 , this is for taking the kth 8 bits in the 32 bit 

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
    }
}