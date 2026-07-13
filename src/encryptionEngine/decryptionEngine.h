#pragma once
#include <iostream>
#include <array>
#include <string>

#include "../chaoticSystems/chenChaoticSystem.h"
#include "../chaoticSystems/lorenzHyperChaoticSystem.h"
#include "imageData.h"

class decryptionEngine{
    private:
        unsigned char* d_scratch_A;
        unsigned char* d_scratch_B;
        unsigned char* d_scratch_C;
        unsigned char* d_scratch_D;
        int* d_permutationMapping;
        unsigned char* d_chaoticMask;

        imageData   m_inputImageDataLayout;
        chenInitialArguments    m_chenInitialArguments;
        lorenzInitialArguments  m_lorenzInitialArguments;
        
        unsigned char* m_chenChaoticStreamRaw;
        unsigned char* m_lorenzChaoticStreamRaw;

    private:
        unsigned char* _chen3DChaoticStreamGeneration();
        unsigned char* _lorenz4DHyperChaoticStreamGeneration();
        void            _allocateDeviceScratchPadData(size_t requriedSize);

        unsigned char* _reverseBitReplacement(unsigned char* input1 , unsigned char* input2 , unsigned char* output , int size);
        unsigned char* _reversePermute(unsigned char* input, int* permutationMap , unsigned char* output , int size);
        unsigned char* _reverseDiffuse(unsigned char* input, unsigned char* diffusionKey ,int width, int height);
        unsigned char* _reverseDNAEncoding(unsigned char* input, unsigned char* ruleKey , unsigned char* output, int size);
        unsigned char* _reverseDNAOperation(unsigned char* input ,unsigned char* chaoticStream, unsigned char* output, int size);
        unsigned char* _reverseImageZipping(unsigned char* input1 , unsigned char* input2 , unsigned char* output, int size);

    public:
        decryptionEngine(const imageData& , const chenInitialArguments& , const lorenzInitialArguments&);
        ~decryptionEngine();

        // Takes BOTH images synchronously and returns the plain text
        unsigned char* decrypt(unsigned char* cipherTextImage, unsigned char* cipherTextImage1, int size);
};