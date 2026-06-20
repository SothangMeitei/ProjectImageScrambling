
#pragma once
#include <cuda_runtime.h>
#include"encryptionEngine.h"

#include "../vendor/stb/stb_image.h"
#include "../vendor/stb/stb_image_write.h"

//kernels for the parallel computations 
        //note that
        //__host__ is purely called by the cpu and executed by the cpu so this is not mentioned
        //__global__ is called by the cpu and then executed by the gpu
        //__device__ is called by the gpu and also executed by the gpu

__global__ void _bitReplaceKernel(
      unsigned char* input
    , unsigned char* output1 
    , unsigned char* output2
    , int            size) {
    
    long long index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        
        unsigned char current_byte = input[index];    
        unsigned char buffer1 = 0;

        //first 4 bits and then the second 4 bits in the second for loop
        for(int i = 0; i < 4; ++i) {
            unsigned char b = (current_byte >> (4 + i)) & 1;
            unsigned char pair = (b << 1) | (b ^ 1);            
            buffer1 |= (pair << (2 * i));
        }
        output1[index] = buffer1;

        unsigned char buffer2 = 0;
        for(int i = 0; i < 4; ++i) {
            unsigned char b = (current_byte >> i) & 1;
            unsigned char pair = (b << 1) | (b ^ 1);
            buffer2 |= (pair << (2 * i));
        }
        
        output2[index] = buffer2;
    }
}

__global__ void _pixelPermuteKernel(
        unsigned char* inputImage
    ,   unsigned char* outputImage
    ,   int* mapping
    ,   int size){

    long long index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index < size){
        //just take the value at the index i in the input and then in the output place it in the mapping[i] position 
        //of the output
        //this thread this responsible for mapping the value 
        //at it's index value in input to the mapping[index] index in the output
        outputImage[mapping[index]] = inputImage[index];
    }
}
__global__ void _pixelDiffuseKernel(unsigned char* input, unsigned char* diffusionMatrix, unsigned char* output , int size){
    long long index = blockIdx.x * blockDim.x + threadIdx.x;
    //this just does the parallel part of the diffusion process the sequential part will be done later in the cpu side
    //as this sequential and parallel parts are associative
    //LATER OPTIMIZE SEQUENTIAL PART OF THIS DIFFUSION PHASE
    if(index < size){
        output[index] = input[index] ^ diffusionMatrix[index];
    }
}

// This lives in the GPU's Constant Memory. 
// It defines the 8 legal Watson-Crick rules.
// Row = Rule (0-7), Column = Binary Value (0-3)
__constant__ int d_DNA_ENCODING_RULES[8][4] = {
    {0, 1, 2, 3}, // Rule 0: A=0, C=1, G=2, T=3
    {0, 2, 1, 3}, // Rule 1: A=0, G=1, C=2, T=3
    {1, 0, 3, 2}, // Rule 2: C=0, A=1, T=2, G=3
    {1, 3, 0, 2}, // Rule 3: C=0, A=2, T=1, G=0
    {3, 1, 2, 0},
    {3, 2, 1, 0},
    {2, 0, 3, 1},
    {2, 3, 0, 1}
};

__device__ int encodeDNA(int binary_val, int rule_index) {
    return d_DNA_ENCODING_RULES[rule_index][binary_val];
}

__global__ void _DNAEncodingKernel(unsigned char* input,unsigned char* mapping , unsigned char* output , int size){
    long long index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < size){
        output[index] = encodeDNA(input[index] ,mapping[index] % 8);   
    }
}
// Used during the Encryption Pipeline
__device__ int performDNA_Addition(int pixel_dna, int key_dna) {
    return (pixel_dna + key_dna) % 4;
}
// Used during the Decryption Pipeline
__device__ int performDNA_Subtraction(int cipher_dna, int key_dna) {
    return (cipher_dna - key_dna + 4) % 4;
}
__global__ void _performDNAOperationKernel(unsigned char* input,unsigned char* chaoticStream ,  unsigned char* output , int size){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < size){
        output[index] = performDNA_Addition(input[index] , chaoticStream[index] );
    }
}

__global__ void _mergeTwoHalvesKernel(unsigned char* input1 , unsigned char* input2 , unsigned char* output , int size){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        // Zipping the two arrays perfectly back into a single 8-bit byte
        output[index] = input1[index] | input2[index];
    }
}



encryptionEngine::encryptionEngine(){
    isRunning = true;
    isPaused = false;
}

std::pair<unsigned char* , unsigned char*> encryptionEngine::_LaunchBitReplace(
      unsigned char* inputImage 
    , int size){
    
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
    _bitReplaceKernel<<<gridSize, blockSize>>>(inputImage, output1, output2, size);
    
    return {output1 , output2};
}
unsigned char* encryptionEngine::_LaunchPixelPermute(
    unsigned char* inputImage
    , int* permutation
    , int  size){
        //this inputImage will be one of the output of the previous bit replacement outputs
        //to not waste the gpu vram since this is not requried in after this point in the pipeline
        //w'll call delete on this inputImage to freeup vram space
        //and in the next point of the pipeline we then just pass the output of this point as the input of the next

        unsigned char* output{nullptr};

        cudaMalloc((void**)&output , size); //this will be the output from this part of the pipeline

        //just to remind we are to take the pixel value in ith position in the input image 
        //and then give to the pixel value in the array[i]th postion in output image position

        //so what kind of computation will a thread do with the index 'index'
        //depending on the index this thread will take the value in that index in the array 
        //then look up the value of the input image at that index and then 
        //assing the value to the output image at the point array[index] 

        int blockSize = 256;
        int gridSize = (size + blockSize - 1) / blockSize;

        _pixelPermuteKernel<<<gridSize , blockSize>>>(inputImage , output , permutation , size);
        return output;
}
unsigned char* encryptionEngine::_LaunchPixelDiffuse(unsigned char* inputImage, unsigned char* diffusionMatrix , int size){
    unsigned char* output{nullptr};
    cudaMalloc((void**)&output , size);

    int blockSize{256};
    int gridSize{(size + blockSize - 1) / blockSize};

    _pixelDiffuseKernel<<<gridSize , blockSize>>>(inputImage , diffusionMatrix , output , size);
    return output;
}
unsigned char* encryptionEngine::_LaunchDNAEncoding(unsigned char* input , unsigned char* keyStream ,  int size){
    //this part just convert the input image into the DNA encodings
    unsigned char* output{nullptr};
    cudaMalloc((void**)&output , size);

    int blockSize{256};
    int gridSize{(size + blockSize - 1) / blockSize};

    _DNAEncodingKernel<<<gridSize , blockSize>>>(input , keyStream , output , size);
    return output;
}
unsigned char* encryptionEngine::_LaunchPerformDNAOperation(unsigned char* input, unsigned char* keyStream, int size){
    //this is for the conversion of the two dna encoded image file with the caotic keystream 
    //in a reversebile manner 
    unsigned char* output{nullptr};
    cudaMalloc((void**)& output , size);

    int blockSize{256};
    int gridSize{(size + blockSize - 1) / blockSize};

    _performDNAOperationKernel(input , keyStream , output , size);
    return output;
}
unsigned char* encryptionEngine::_LauchImageMerginZip(unsigned char* input1 , unsigned char* input2 , int size){
    unsigned char* output{nullptr};
    cudaMalloc((void**)&output , size);

    int blockSize{256};
    int gridSize{(size + blockSize - 1) / blockSize};

    _mergeTwoHalvesKernel(input1 , input2 , output , size);
    return output;
}
unsigned char* encryptionEngine::_LaunchDNADecoding(unsigned char * input , int size){}
unsigned char* encryptionEngine::_encrypt(unsigned char* plainTextInputImage , int size){
    //get the input data
    //malloc the new place where the output is to be stored
    //give the pointer to the new data and the input data to the kernel
    //the kernel then populates the new memory location in vram
    //then this pointer to the output will be pointing to the new memory location with the encrypted image

    //the size of the image will be a standard 1080p , with 16 : 9 aspect ratio that is standard HD
    //or for 4k the size is double that (not exactly) , but this is be a variable input into the size  variable
    
    std::pair outputsBitReplacement = _LaunchBitReplace(plainTextInputImage , size);

    auto outputPermutedEncoding1 = _LaunchPixelPermute(outputsBitReplacement.first , nullptr , size);
    auto outputPermutedEncoding2 = _LaunchPixelPermute(outputsBitReplacement.second , nullptr , size);

    auto outputDiffusedEncoding1 = _LaunchPixelDiffuse(outputPermutedEncoding1 , nullptr , size);
    auto outputDiffusedEncoding2 = _LaunchPixelDiffuse(outputPermutedEncoding2 , nullptr , size);

    auto outputDNAEncodedImage1 = _LaunchDNAEncoding(outputDiffusedEncoding1  , nullptr , size);
    auto outputDNAEncodedImage2 = _LaunchDNAEncoding(outputDiffusedEncoding2  , nullptr , size);

    auto outputDNAOperatedMergedImage = _LaunchPerformDNAOperation(outputDNAEncodedImage1 , outputDNAEncodedImage2 , size);

    auto outputDNADecodedFinalImage = _LaunchDNADecoding(outputDNAOperatedMergedImage , size);

    return outputDNADecodedFinalImage;
}
unsigned char* encryptionEngine::_decrypt(unsigned char* cypherTextImage , int size){
    //thisis to get in the input as the encrypted text file and then run the decryption algorithm 
    //on that input cypher text file
}

bool encryptionEngine::pushImageIntoQueueBuffer(const unsigned char* input){
    //the input is pushed into this buffer this will then pop the latest available image when it could be processed
    //this input already exist in the ram and this function is just to make this go into the processing queue

}

void encryptionEngine::run(){
    while(isRunning){
        
    }
}
