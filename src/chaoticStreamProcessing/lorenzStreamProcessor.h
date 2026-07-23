#pragma once
#include <cstdint>

class lorenzStreamProcessor {
private:
    int       m_size;      // Target pixel/byte count
    uint32_t* m_intStream; // Contiguous 32-bit diffusion stream

public:
    lorenzStreamProcessor(int pixelCount);
    ~lorenzStreamProcessor();

    uint32_t extractIntegralValues(double floatingValue);
    void     ingestRawStream(const double* rawLorenzStream);

    uint32_t* getDiffusionValues() const { return m_intStream; }
};