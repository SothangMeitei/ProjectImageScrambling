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

class encryptionEngine{
    private:
        //flags
        bool isRunning;
        bool isPaused;

        int                     m_streamSize;
        imageData               m_referenceFormat;
        chenInitialArguments    m_chenArguments;
        lorenzInitialArguments  m_lorenzArguments;

    private:
        //internal data structures
        std::queue<imageData>  m_inputImageQueueBuffer;
        unsigned char * m_chaoticStreamChen;
        unsigned char * m_chaoticStreamLorenz;

        int* d_permMap;
    private:
        // --- THE VRAM SCRATCHPAD ARENA ---
        unsigned char* d_scratchA;
        unsigned char* d_scratchB;
        unsigned char* d_scratchC;
        unsigned char* d_scratchD;
        
        size_t m_currentArenaPixelSize; // Tracks the size the arena was built for
        void _reallocateVRAMScratchpadIfNeeded(size_t required_size);

    private:
        //for the cpu side computation

        //populate the initial paramters
        void setParamChen(std::array<double , 3>);
        void setParamLorez(std::array<double , 5>);
        //the key is probably generated using the pseudo random generator using the computer system or something
        //this populates the Starting parameter internal member varialbe for both the coatic system
        void secreatKeyGenerator();
        //this is the chaotic stream generators that will operate solely in the cpu side
        unsigned char* chen3DChaoticStream();
        unsigned char* lorenz4DHyperChaoticStream();

    private:
        //functions or systems that will work on the image files and the image files as datas
        //private as this is to be done only on the image files inside this class only and not on something outside of this class
        //here bool is used to indicate where we are doing the encrption or the decrption process
        std::pair<unsigned char* , unsigned char*> _LaunchBitReplace(unsigned char* ,unsigned char* , unsigned char*,  int);
        //all of which is just the input , output , size ; in that order in this generation of the next stage in the pipeline the same vram location may be reused
        unsigned char* _LaunchPixelPermute(unsigned char* , unsigned char* , int* , int);
        unsigned char* _LaunchPixelDiffuse(unsigned char* , unsigned char* ,unsigned char* , int);
        unsigned char* _LaunchDNAEncoding(unsigned char* , unsigned char* , unsigned char* ,int);
        unsigned char* _LaunchPerformDNAOperation(unsigned char* , unsigned char* , unsigned char* ,int);
        unsigned char* _LaunchDNADecoding(unsigned char * ,unsigned char * ,  int);
        unsigned char* _LauchImageMerginZip(unsigned char*, unsigned char*, unsigned char* , int);

        unsigned char* _encrypt(unsigned char* , int size);
    public:
        encryptionEngine(const imageData& ,const chenInitialArguments& ,const lorenzInitialArguments&);
        ~encryptionEngine();

        bool pushImageIntoQueueBuffer(const unsigned char*);
        int getRemainingQueueSize(){return m_inputImageQueueBuffer.size();}

        void run();
        void stop();
};