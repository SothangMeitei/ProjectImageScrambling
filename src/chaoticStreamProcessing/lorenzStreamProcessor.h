#pragma once
#include <cstdint>
#include "../chaoticSystems/lorenzHyperChaoticSystem.h" // Add include

class lorenzStreamProcessor {
private:
    int       m_size;      // Target pixel/byte count
    uint32_t* m_intStream; // Contiguous 32-bit diffusion stream
public:
    lorenzStreamProcessor(int pixelCount);
    ~lorenzStreamProcessor();
    
    // Ingest the full 4D struct instead of const double*
    void ingestRawStream(const chaoticStreamLorenz<double>& stream); 
    
    uint32_t* getDiffusionValues() const { return m_intStream; }
};