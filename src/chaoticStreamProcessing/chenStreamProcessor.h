#pragma once
#include <cstdint>
#include "../chaoticSystems/chenChaoticSystem.h" // Add include

class chenStreamProcessor {
public:
    struct mappingArrayValue {
        uint64_t mantissaChaos; 
        int      previousIndex; 
    };
private:
    int                m_size;
    int*               m_flatMapping; 
    mappingArrayValue* m_structArray; 
    unsigned char*     m_byteStream;  // NEW: Aligns with lorenzStreamProcessor's design
    void _radixSort();
public:
    chenStreamProcessor(int streamSize);
    ~chenStreamProcessor();
    
    // Ingest the full 3D struct
    void ingestRawStream(const chaoticStreamChen<double>& stream);
    void sortAndExtractMapping();
    
    int* getGPUFlatMapping() const { return m_flatMapping; }
    unsigned char* getByteStream() const { return m_byteStream; } // NEW: Getter for DNA Rules
};