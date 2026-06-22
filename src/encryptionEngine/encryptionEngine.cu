
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
__global__ void _pixelDiffuseKernel(
    unsigned char* input
    , unsigned char* diffusionMatrix
    , unsigned char* output
    , int size){
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

__device__ int encodeDNA(
    int binary_val
    , int rule_index) {
    return d_DNA_ENCODING_RULES[rule_index][binary_val];
}

__global__ void _DNAEncodingKernel(
    unsigned char* input
    , unsigned char* mapping
    , unsigned char* output
    , int size){
    long long index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < size){
        output[index] = encodeDNA(input[index] ,mapping[index] % 8);   
    }
}
// Used during the Encryption Pipeline
__device__ int performDNA_Addition(
    int pixel_dna
    , int key_dna) {
    return (pixel_dna + key_dna) % 4;
}
// Used during the Decryption Pipeline
__device__ int performDNA_Subtraction(
    int cipher_dna
    , int key_dna) {
    return (cipher_dna - key_dna + 4) % 4;
}
__global__ void _performDNAOperationKernel(
    unsigned char* input
    , unsigned char* chaoticStream
    , unsigned char* output , int size){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < size){
        output[index] = performDNA_Addition(input[index] , chaoticStream[index] );
    }
}
__global__ void _mergeTwoHalvesKernel(
    unsigned char* input1
    , unsigned char* input2 
    , unsigned char* output
    , int size){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        // Zipping the two arrays perfectly back into a single 8-bit byte
        output[index] = input1[index] | input2[index];
    }
}
__global__ void _performDNADecoding(
    unsigned char* input1
    , unsigned char* output
    , int size){}
encryptionEngine::encryptionEngine(){
    isRunning = true;
    isPaused = false;

    d_scratchA = nullptr;
    d_scratchB = nullptr;
    d_scratchC = nullptr;
    d_scratchD = nullptr;
    m_currentArenaPixelSize = 0;
}

std::pair<unsigned char* , unsigned char*> encryptionEngine::_LaunchBitReplace(
    unsigned char* inputImage
    , unsigned char* output1
    , unsigned char* output2
    , int size){
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
    , unsigned char* output
    , int* permutation
    , int  size){
        //this inputImage will be one of the output of the previous bit replacement outputs
        //to not waste the gpu vram since this is not requried in after this point in the pipeline
        //w'll call delete on this inputImage to freeup vram space
        //and in the next point of the pipeline we then just pass the output of this point as the input of the next

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
unsigned char* encryptionEngine::_LaunchPixelDiffuse(
    unsigned char* inputImage
    , unsigned char* diffusionMatrix
    , unsigned char* output
    , int size){

    int blockSize{256};
    int gridSize{(size + blockSize - 1) / blockSize};

    _pixelDiffuseKernel<<<gridSize , blockSize>>>(inputImage , diffusionMatrix , output , size);
    return output;
}
unsigned char* encryptionEngine::_LaunchDNAEncoding(
    unsigned char* input
    , unsigned char* keyStream
    , unsigned char* output
    , int size){
    //this part just convert the input image into the DNA encodings
    int blockSize{256};
    int gridSize{(size + blockSize - 1) / blockSize};

    _DNAEncodingKernel<<<gridSize , blockSize>>>(input , keyStream , output , size);
    return output;
}
unsigned char* encryptionEngine::_LaunchPerformDNAOperation(
    unsigned char* input
    , unsigned char* keyStream
    , unsigned char* output
    , int size){
    //this is for the conversion of the two dna encoded image file with the caotic keystream 
    //in a reversebile manner
    int blockSize{256};
    int gridSize{(size + blockSize - 1) / blockSize};

    _performDNAOperationKernel(input , keyStream , output , size);
    return output;
}
unsigned char* encryptionEngine::_LauchImageMerginZip(
    unsigned char* input1 
    , unsigned char* input2 
    , unsigned char* output
    , int size){

    int blockSize{256};
    int gridSize{(size + blockSize - 1) / blockSize};

    _mergeTwoHalvesKernel(input1 , input2 , output , size);
    return output;
}
unsigned char* encryptionEngine::_LaunchDNADecoding(
    unsigned char* input
    , unsigned char* output
    , int size){
    
    int blockSize{256};
    int gridSize{(size + blockSize - 1) / blockSize};

    _performDNADecoding<<<blockSize , gridSize>>>(input , output , size);
    return output;
}

void encryptionEngine::_reallocateVRAMScratchpadIfNeeded(size_t required_size) {
    if (required_size > m_currentArenaPixelSize) {
        if(d_scratchA) { cudaFree(d_scratchA); cudaFree(d_scratchB); cudaFree(d_scratchC); cudaFree(d_scratchD); }
        
        cudaMalloc((void**)&d_scratchA, required_size);
        cudaMalloc((void**)&d_scratchB, required_size);
        cudaMalloc((void**)&d_scratchC, required_size);
        cudaMalloc((void**)&d_scratchD, required_size);
        
        m_currentArenaPixelSize = required_size;
    }
}
unsigned char* encryptionEngine::_encrypt(unsigned char* plainTextInputImage, int size) {
    
    // 1. Guarantee the VRAM scratchpad is large enough for this frame
    _reallocateVRAMScratchpadIfNeeded(size);

    // Copy input image to VRAM (We can use ScratchB temporarily as our landing pad!)
    cudaMemcpy(d_scratchB, plainTextInputImage, size, cudaMemcpyHostToDevice);

    // ------------------------------------------------------------------------
    // THE PING-PONG PIPELINE (Zero Driver Traps, 100% Asynchronous Silicon Speed)
    // ------------------------------------------------------------------------
    
    // Stage 1: Split ScratchB ---> writes into ScratchA (MSB) and ScratchC (LSB)
    _LaunchBitReplace(d_scratchB, d_scratchA, d_scratchC, size);

    // Stage 2: Permutation
    // MSB: Reads ScratchA ---> writes to ScratchB
    _LaunchPixelPermute(d_scratchA, d_scratchB, d_permMap, size);
    // LSB: Reads ScratchC ---> writes to ScratchD
    _LaunchPixelPermute(d_scratchC, d_scratchD, d_permMap, size);

    // Stage 3: Diffusion
    // MSB: Reads ScratchB ---> writes back to ScratchA
    _LaunchPixelDiffuse(d_scratchB, chaoticStreamLorenz, d_scratchA, size);
    // LSB: Reads ScratchD ---> writes back to ScratchC
    _LaunchPixelDiffuse(d_scratchD, chaoticStreamLorenz, d_scratchC, size);

    // Stage 4: DNA Encoding
    // MSB: Reads ScratchA ---> writes to ScratchB
    _LaunchDNAEncoding(d_scratchA, chaoticStreamLorenz, d_scratchB, size);
    // LSB: Reads ScratchC ---> writes to ScratchD
    _LaunchDNAEncoding(d_scratchC, chaoticStreamLorenz, d_scratchD, size);

    // Stage 5: Reversible DNA Merge (Merge LSB into MSB stream)
    // Reads ScratchB & ScratchD ---> writes final answer into ScratchA
    _LaunchPerformDNAOperation(d_scratchB, d_scratchD, d_scratchA, size);

    // ... Decode DNA from ScratchA into ScratchB ...
    _LaunchDNADecoding(d_scratchA , d_scratchB , size);
    

    // Copy the final encrypted result from ScratchB back to CPU RAM
    unsigned char* h_encrypted_output = new unsigned char[size];
    cudaMemcpy(h_encrypted_output, d_scratchB, size, cudaMemcpyDeviceToHost);

    return h_encrypted_output;
}
unsigned char* encryptionEngine::_decrypt(unsigned char* cypherTextImage , int size){
    //thisis to get in the input as the encrypted text file and then run the decryption algorithm 
    //on that input cypher text file
}

bool encryptionEngine::pushImageIntoQueueBuffer(const unsigned char* input){
    //the input is pushed into this buffer this will then pop the latest available image when it could be processed
    //this input already exist in the ram and this function is just to make this go into the processing queue

}
void encryptionEngine::stop(){
    isRunning = false;
}

void encryptionEngine::run(){
    while(isRunning){
        //process the queue
        //and then write the output to the disk and then after that
        //process the next frame in the queue
        if(m_inputImageQueueBuffer.empty()) continue;
        
        unsigned char* cipherText{_encrypt(
            m_inputImageQueueBuffer.front().imagePixelValues 
        ,   m_inputImageQueueBuffer.front().sizeOfImageFileInByte)};

        m_inputImageQueueBuffer.pop();
        
        //need to write this into the disk some how save this cipher text in the disk dont know how to do this
    }
}
