#pragma once
#include <cstdint>

class lorenzStreamProcessor {
    private:
        int                     m_size;
        uint32_t*                    m_intStream;  //requried final stream of int to be added to the image file
    private:
        void _floatToInt();
    public:
        lorenzStreamProcessor(int streamSize);
        ~lorenzStreamProcessor();

        void ingestRawStream(const float* rawLorenzStream);
        uint32_t extractIntegralValues(float);

        uint32_t* getDiffusionValues() const { return m_intStream; }
};