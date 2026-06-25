

#pragma once
#include<iostream>
#include<queue>
#include<vector>
#include<unordered_map>
#include<functional>
#include<string>
#include<array>

#include"../chaoticSystems/chenChaoticSystem.h"
#include"../chaoticSystems/lorenzHyperChaoticSystem.h"

struct imageData{
    unsigned char* imagePixelValues;
    int sizeOfImageFileInByte;
    int height;
    int width;
    int channels;
};

class decryptionEngine{
    private:
        //internal flags
        bool isRunning;
        bool isPaused;

    private:
        std::queue<imageData> m_decryptionQueue;
        //data sturcture pointer for the data living in the gpu as scratch pad
        unsigned char* d_scratch_A;
        unsigned char* d_scratch_B;
        unsigned char* d_scratch_C;
        unsigned char* d_scratch_D;
        int*           d_permutationMapping;
        unsigned char* d_chaoticMask;

        imageData   m_inputImageDataLayout;
        
    private:
        chenInitialArguments    m_chenInitialArguments;
        lorenzInitialArguments  m_lorenzInitialArguments;
        //this are the pure raw chaotic stream
        unsigned char*   m_chenChaoticStreamRaw;
        unsigned char*   m_lorenzChaoticStreamRaw;
        //this is the processed mapping after the process of sorting on the raw chenStream
    private:
        //population functions
        unsigned char*  _chen3DChaoticStreamGeneration();
        unsigned char*  _lorenz4DHyperChaoticStreamGeneration();
        void            _allocateDeviceScratchPadData(size_t requriedSize);
    private:
        //internal decryption functions
        unsigned char* _reversePermute(unsigned char* input, int* permutationMap , unsigned char* output , int size);
        unsigned char* _reverseDiffuse(unsigned char* input, unsigned char* diffusionKey , unsigned char* output, int size);
        unsigned char* _reverseDNAEncoding(unsigned char* input, unsigned char* ruleKey , unsigned char* output, int size);
        unsigned char* _reverseDNAOperation(unsigned char* input ,unsigned char* chaoticStream, unsigned char* output, int size);
        unsigned char* _reverseImageZipping(unsigned char* input1 , unsigned char* input2 , unsigned char* output, int size);

        unsigned char* _decrypt(unsigned char* cipherText, int size);

    public:
        decryptionEngine(const imageData& , const chenInitialArguments& , const lorenzInitialArguments&);
        ~decryptionEngine();
        //this will return false on faliur
        bool pushIntoTheQueue(imageData&);
        void run();
        void stop(){isRunning = false;}
        void pause(){isPaused = true;}
        void resume(){isPaused = false;}
};