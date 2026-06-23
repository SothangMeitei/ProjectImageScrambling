
#pragma once
#include <cstdint>

class lorenzStreamProcessor {
    private:
        int m_size;                  // Target pixel count in bytes
        unsigned char* m_byteStream; // Contiguous 8-bit stream for VRAM

    public:
        lorenzStreamProcessor(int pixelCount);
        ~lorenzStreamProcessor();

        void ingestRawStream(const float* rawLorenzStream, int floatCount);
        unsigned char* getDiffusionBytes() const { return m_byteStream; }
};