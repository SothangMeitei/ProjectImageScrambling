#pragma once
#include <cstdint>

class chenStreamProcessor {
public:
    struct mappingArrayValue {
        uint64_t mantissaChaos; // 64-bit scaled fractional key (frac * 1e14)
        int      previousIndex; // Original spatial index (0 to size-1)
    };

private:
    int                m_size;
    int*               m_flatMapping; // Flat mapping array for GPU
    mappingArrayValue* m_structArray; // Struct array for Radix sorting

    void _radixSort();

public:
    chenStreamProcessor(int streamSize);
    ~chenStreamProcessor();

    // Ingests 64-bit double stream from chenChaoticSystem<double>
    void ingestRawStream(const double* rawChenStream);
    void sortAndExtractMapping();

    int* getGPUFlatMapping() const { return m_flatMapping; }
};