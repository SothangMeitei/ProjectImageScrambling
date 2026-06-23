
#pragma once
#include <cstdint>

class lorenzStreamProcessor {
    private:
        int m_size;                  // Target pixel count in bytes
        uint32_t* m_intStream; // Contiguous 8-bit stream for VRAM

    public:
        lorenzStreamProcessor(int pixelCount);
        ~lorenzStreamProcessor();

        uint32_t extractIntegralValues(float floatingValue);
        void ingestRawStream(const float* rawLorenzStream);
        uint32_t* getDiffusionValues() const { return m_intStream; }
};