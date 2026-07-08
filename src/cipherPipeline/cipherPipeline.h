#pragma once
#include<functional>
#include<vector>

class cipherPipeline{
    private:
        std::vector<std::function<void(unsigned char* input , unsigned char* output , unsigned char* keystream , int size)>> functionStream;
        
    public:
};