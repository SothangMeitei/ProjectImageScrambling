#pragma once
//define all the functions for the chen chaotic stream processing

//according to the engine design we need to have the mapping array of int
//that is the mapping[index] = new position , that is 
//for old position = index 
//new position = mapping[index]

#pragma once
#include <cstdint>

class chenStreamProcessor {
    public:
        struct mappingArrayValue {
            uint64_t mantissaChaos; // The 52-bit chaotic fraction cast to integer
            int previousIndex;      // The original spatial index (0 to size-1)
        };

    private:
        int                     m_size;
        int* m_flatMapping;                     // The flat array for the GPU
        mappingArrayValue* m_structArray;       // The structural array for sorting
    private:
        void _radixSort(int , unsigned char*);
    public:
        chenStreamProcessor(int streamSize);
        ~chenStreamProcessor();

        void ingestRawStream(const float* rawChenStream);
        void sortAndExtractMapping();

        int* getGPUFlatMapping() const { return m_flatMapping; }
};