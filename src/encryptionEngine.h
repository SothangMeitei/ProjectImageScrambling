#pragma once
#include<iostream>
#include<queue>
#include<vector>
#include<unordered_map>
#include<functional>
#include<string>

class encryptionEngine{
    private:
        //flags
        bool isRunning;
        bool isPaused;
    private:
        std::queue<unsigned char*> m_inputImageQueueBuffer;
        std::unordered_map<std::string , std::function<void()>> m_DNAOperationMap;

        enum class DNA{
            A,G,T,C     //could use this ordering as the binary representation instead of having 
        };              //the expicite mapping but the mapping is more powerfull and better at customizabitlity
        std::unordered_map<DNA , int> m_DNAMappingFromDNAToBinaryInteger;
    
    private:
        //kernels for the parallel computations 
        //note that
        //__host__ is purely called by the cpu and executed by the cpu so this is not mentioned
        //__global__ is called by the cpu and then executed by the gpu
        //__device__ is called by the gpu and also executed by the gpu

        __global__ _bitReplace8To16(unsigned char*);
    private:
        //functions or systems that will work on the image files and the image files as datas
        //private as this is to be done only on the image files inside this class only and not on something outside of this class
        //here bool is used to indicate where we are doing the encrption or the decrption process
        void _bitReplace(unsigned char* , bool);
        void _pixelPermute(unsigned char* , const std::vector<long long>& , bool);
        void _pixelDiffuse(unsigned char* , unsigned char* , bool);
        void _DNAEncoding(unsigned char* , bool);
        void _performDNAOperation(unsigned char* , const std::string& , bool);
        void _DNADecoding(unsigned char* , bool);

        void _encrypt(unsigned char*);
        void _decrypt(unsigned char*);

    public:
        encryptionEngine();

        bool pushImageIntoQueueBuffer(const unsigned char*);

        void run();
        void pause();
};