
#pragma once
#include <cuda_runtime.h>
#include"encryptionEngine.h"

#include "../vendor/stb/stb_image.h"
#include "../vendor/stb/stb_image_write.h"

encryptionEngine::encryptionEngine(){
    isRunning = true;
    isPaused = false;
}
__global__ void _bitReplaceKernel(
      unsigned char* input
    , unsigned char* output1 
    , unsigned char* output2
    , long long size
    , std::array<char , 2> mappingData
    , bool isEncrypt){
    //for all bits 
    //if 1 then make 10 and give 1 to output1 and 0 to output2
    //if 0 then make 01 and give 0 to output1 and 1 to output2
    // Calculate the 1D index
    long long index = blockIdx.x * blockDim.x + threadIdx.x;

    // Safety boundary check
    if (index < size) {
        // Read the byte
        unsigned char current_byte = input[index];
        unsigned char outputByteBuffer{0};
        
        for(int i = 3; i >= 0 ; ++i){
            if(((current_byte >> 4 + i) && 1) == 0){
                //if 0 then
                outputByteBuffer || (1 << (2 * i));
            }
            else{
                outputByteBuffer || (1 << (2 * i + 1));
            }
            output1[index] = outputByteBuffer;
        }

        outputByteBuffer = 0;

        for(int i = 3; i >= 0 ; ++i){
            if(((current_byte >> i) && 1) == 0){
                //if 0 then
                outputByteBuffer || (1 << (2 * i));
            }
            else{
                outputByteBuffer || (1 << (2 * i + 1));
            }
            output2[index] = outputByteBuffer;
        }
    }
}

__global__ void _pixelPermuteKernel(
    unsigned char* inputImage 
    , long long* mapping
    , bool isReverse){
    //this mapping is already sorted and then this is to be used for the purpose of mapping the 
    //new indices for the pixels in the position after the 1D has been resized into a 2D structure
    if(isReverse){
        //reverse the permutation
        // confuse -> plaintext
    }
    else{
        // plaintext -> confused 
    }
}
__global__ void _pixelDiffuseKernel(unsigned char* , unsigned char* , bool){}
__global__ void _DNAEncodingKernel(unsigned char* , bool){}
__global__ void _performDNAOperationKernel(unsigned char* , const std::string&){}


std::pair<unsigned char* , unsigned char*> encryptionEngine::_LaunchBitReplace(
      unsigned char* inputImage 
    , long long size 
    , std::array<char , 2> bitMap
    , bool isEncrypt){
    
    unsigned char* output1{nullptr};
    unsigned char* output2{nullptr};

    cudaMalloc((void**)&output1 , size);     //size here is the size of the memory that is to be allocated in the memory
    cudaMalloc((void**)&output2 , size);

    //this means that there is 256 threads per block 
    int blockSize = 256; 

    // A flat, 1D grid. 
    // The "+ blockSize - 1" is a math trick to ensure if your total size isn't 
    // perfectly divisible by 256, it spawns one extra block to catch the leftovers.
    int gridSize = (size + blockSize - 1) / blockSize; 

    // 3. The Launch Syntax (Note: Grid ALWAYS comes first!)
    _bitReplaceKernel<<<gridSize, blockSize>>>(inputImage, output1, output2, size, bitMap, isEncrypt);
    
    return {output1 , output2};
}
unsigned char* encryptionEngine::_LaunchPixelPermute(unsigned char* , const std::vector<long long>& , bool){}
unsigned char* encryptionEngine::_LaunchPixelDiffuse(unsigned char* , unsigned char* , bool){}
unsigned char* encryptionEngine::_LaunchDNAEncoding(unsigned char* , bool){}
unsigned char* encryptionEngine::_LaunchPerformDNAOperation(unsigned char* , const std::string&){}

void encryptionEngine::_encrypt(unsigned char*){
    //get the input data
    //malloc the new place where the output is to be stored
    //give the pointer to the new data and the input data to the kernel
    //the kernel then populates the new memory location in vram
    //then this pointer to the output will be pointing to the new memory location with the encrypted image

}
void encryptionEngine::_decrypt(unsigned char*){}


bool encryptionEngine::pushImageIntoQueueBuffer(const unsigned char*){}

void encryptionEngine::run(){}
void encryptionEngine::pause(){}