#pragma once
#include <iostream>
#include <array>
#include <string>
#include <utility> // For std::pair

#include "../chaoticSystems/chenChaoticSystem.h"
#include "../chaoticSystems/lorenzHyperChaoticSystem.h"
#include "imageData.h"

class encryptionEngine{
    private:
        int                     m_streamSize;
        imageData               m_referenceFormat;
        chenInitialArguments    m_chenArguments;
        lorenzInitialArguments  m_lorenzArguments;

    private:
        unsigned char * m_chaoticStreamChen;
        unsigned char * m_chaoticStreamLorenz;
        int* d_permMap;

    private:
        // --- THE VRAM SCRATCHPAD ARENA ---
        unsigned char* d_scratchA;
        unsigned char* d_scratchB;
        unsigned char* d_scratchC;
        unsigned char* d_scratchD;
        
        size_t m_currentArenaPixelSize; 
        void _reallocateVRAMScratchpadIfNeeded(size_t required_size);

    private:
        unsigned char* chen3DChaoticStream();
        unsigned char* lorenz4DHyperChaoticStream();

        std::pair<unsigned char* , unsigned char*> _LaunchBitReplace(unsigned char* ,unsigned char* , unsigned char*,  int);
        unsigned char* _LaunchPixelPermute(unsigned char* , unsigned char* , int* , int);
        unsigned char* _LaunchPixelDiffusion(unsigned char* , unsigned char* , int  ,int);
        unsigned char* _LaunchDNAEncoding(unsigned char* , unsigned char* , unsigned char* ,int);
        unsigned char* _LaunchPerformDNAOperation(unsigned char* , unsigned char* , unsigned char* ,int);
        unsigned char* _LauchImageMerginZip(unsigned char*, unsigned char*, unsigned char* , int);

    public:
        encryptionEngine(const imageData& ,const chenInitialArguments& ,const lorenzInitialArguments&);
        ~encryptionEngine();

        // Now perfectly synchronous and returns BOTH images
        std::pair<unsigned char*, unsigned char*> encrypt(unsigned char* plainTextInputImage, int size);
};