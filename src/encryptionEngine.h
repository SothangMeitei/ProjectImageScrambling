#pragma once
#include<iostream>
#include<queue>
#include<vector>
#include<unordered_map>
#include<functional>
#include<string>
#include<array>

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
    private:
        //functions or systems that will work on the image files and the image files as datas
        //private as this is to be done only on the image files inside this class only and not on something outside of this class
        //here bool is used to indicate where we are doing the encrption or the decrption process
        std::pair<unsigned char* , unsigned char*> _LaunchBitReplace(unsigned char* ,long long ,std::array<char , 2>, bool);
        unsigned char* _LaunchPixelPermute(unsigned char* , const std::vector<long long>& , bool);
        unsigned char* _LaunchPixelDiffuse(unsigned char* , unsigned char* , bool);
        unsigned char* _LaunchDNAEncoding(unsigned char* , bool);
        unsigned char* _LaunchPerformDNAOperation(unsigned char* , const std::string&);

        void _encrypt(unsigned char*);
        void _decrypt(unsigned char*);

    public:
        encryptionEngine();

        bool pushImageIntoQueueBuffer(const unsigned char*);

        void run();
        void pause();
};