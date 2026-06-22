#include"lorenzStreamProcessor.h"
#include <bit>
#include <cstdint>

void lorenzStreamProcessor::_floatToInt(){}

lorenzStreamProcessor::lorenzStreamProcessor(int streamSize) 
: m_size{streamSize} {
    m_intStream = new uint32_t[m_size];
}
lorenzStreamProcessor::~lorenzStreamProcessor(){
    delete [] m_intStream;
}

void lorenzStreamProcessor::ingestRawStream(const float* rawLorenzStream){
    for(int i = 0 ; i < m_size ; ++i){
        m_intStream[i] = extractIntegralValues(rawLorenzStream[i]);
    }
}

uint32_t lorenzStreamProcessor::extractIntegralValues(float floatingValue)
{
    uint32_t bits = std::bit_cast<uint32_t>(floatingValue);
    return bits & 0xFFFFFF; // extract lowest 24 bits
}
