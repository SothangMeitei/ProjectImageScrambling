
#pragma once

#include"decryptionEngine.h"
#include <cuda_runtime.h>
#include "encryptionEngine.h"

#include "../../vendor/stb/stb_image.h"
#include "../../vendor/stb/stb_image_write.h"
#include <filesystem>

#include<cstring>
#include"../chaoticStreamProcessing/chenStreamProcessor.h"
#include"../chaoticStreamProcessing/lorenzStreamProcessor.h"

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>

unsigned char* decryptionEngine::_chen3DChaoticStreamGeneration() {
    int size = m_inputImageDataLayout.sizeOfImageFileInByte;

    chenChaoticSystem chenSolver(m_chenInitialArguments, size);
    chenSolver.generate();
    chaoticStreamChen rawChen = chenSolver.getChaoticStreams();

    chenStreamProcessor permProcessor(size);
    permProcessor.ingestRawStream(rawChen.x);
    permProcessor.sortAndExtractMapping();

    cudaMalloc((void**)&d_permutationMapping, size * sizeof(int));
    cudaMemcpy(d_permutationMapping, permProcessor.getGPUFlatMapping(), size * sizeof(int), cudaMemcpyHostToDevice);

    // MINT PURE 8-BIT NOISE (Removes the & 7 underflow trap!)
    unsigned char* h_dnaRules = new unsigned char[size];
    for (int i = 0; i < size; ++i) {
        uint32_t bits;
        std::memcpy(&bits, &rawChen.y[i], sizeof(float));
        
        bits ^= bits >> 16;
        bits *= 0x85ebca6b;
        bits ^= bits >> 13;
        bits *= 0xc2b2ae35;
        bits ^= bits >> 16;
        
        h_dnaRules[i] = (unsigned char)(bits & 0xFF); 
    }

    unsigned char* d_dnaRulesDevice = nullptr;
    cudaMalloc((void**)&d_dnaRulesDevice, size);
    cudaMemcpy(d_dnaRulesDevice, h_dnaRules, size, cudaMemcpyHostToDevice);

    delete[] h_dnaRules;
    return d_dnaRulesDevice; 
}
unsigned char* decryptionEngine::_lorenz4DHyperChaoticStreamGeneration() {
    size_t totalPayloadBytes = m_inputImageDataLayout.sizeOfImageFileInByte;
    size_t requiredIntWords = (totalPayloadBytes + 3) / 4;

    lorenzChaoticSystem lorenzSolver(m_lorenzInitialArguments, requiredIntWords);
    lorenzSolver.generate();

    lorenzStreamProcessor mint(requiredIntWords);
    mint.ingestRawStream(lorenzSolver.getChaoticStream().x);
    uint32_t* cpuIntReservoir = mint.getDiffusionValues();

    unsigned char* d_vramByteStream = nullptr;
    cudaMalloc((void**)&d_vramByteStream, totalPayloadBytes);
    cudaMemcpy(d_vramByteStream, cpuIntReservoir, totalPayloadBytes, cudaMemcpyHostToDevice);

    return d_vramByteStream; 
}

void decryptionEngine::_allocateDeviceScratchPadData(size_t sizeOfScratchPad){
    if(d_scratch_A == nullptr){
        cudaMalloc((void**)&d_scratch_A , sizeOfScratchPad);
        cudaMalloc((void**)&d_scratch_B , sizeOfScratchPad);
        cudaMalloc((void**)&d_scratch_C , sizeOfScratchPad);
        cudaMalloc((void**)&d_scratch_D , sizeOfScratchPad);
    }
}




//================================================================================================================================================

//helper functions for the forward encryption phase 
//this is copy pasted form the other encryption part , same as the encryption part of the engine
__global__ void _pixelDiffuseKernel(unsigned char* input, unsigned char* diffusionMatrix, unsigned char* output, int size) {
    long long index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < size) {
        output[index] = input[index] ^ diffusionMatrix[index];
    }
}

__constant__ int d_DNA_ENCODING_RULES[8][4] = {
    {0, 1, 2, 3}, {0, 2, 1, 3}, {1, 0, 3, 2}, {1, 3, 0, 2},
    {3, 1, 2, 0}, {3, 2, 1, 0}, {2, 0, 3, 1}, {2, 3, 0, 1}
};

__device__ int encodeDNA(int binary_val, int rule_index) {
    return d_DNA_ENCODING_RULES[rule_index][binary_val];
}

__global__ void _DNAEncodingKernel(unsigned char* input, unsigned char* mapping, unsigned char* output, int size) {
    long long index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < size) {
        unsigned char current_byte = input[index];
        int rule = mapping[index] % 8; // Safely restrains the 8-bit noise to 0-7 dynamically
        unsigned char buffer = 0;
        for (int b = 0; b < 4; ++b) {
            int base_val = (current_byte >> (2 * b)) & 3; 
            int encoded_base = encodeDNA(base_val, rule);
            buffer |= (encoded_base << (2 * b));
        }
        output[index] = buffer;
    }
}

__device__ int performDNA_Addition(int pixel_dna, int key_dna) {
    return (pixel_dna + key_dna) % 4;
}

__global__ void _performDNAOperationKernel(unsigned char* input, unsigned char* chaoticStream, unsigned char* output, int size) {
    long long index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < size) {
        unsigned char img_byte = input[index];
        unsigned char key_byte = chaoticStream[index];

        unsigned char result_byte = 0;
        for (int b = 0; b < 4; ++b) {
            int img_base = (img_byte >> (2 * b)) & 3;
            int key_base = (key_byte >> (2 * b)) & 3;
            int added_base = performDNA_Addition(img_base, key_base);
            result_byte |= (added_base << (2 * b));
        }
        output[index] = result_byte;
    }
}
unsigned char* _LaunchPixelDiffuse(
    unsigned char* inputImage, unsigned char* diffusionMatrix, unsigned char* output, int size) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    _pixelDiffuseKernel<<<gridSize, blockSize>>>(inputImage, diffusionMatrix, output, size);

    // O(log N) Inclusive Prefix XOR Scan natively in VRAM
    thrust::device_ptr<unsigned char> dev_ptr(output);
    thrust::inclusive_scan(dev_ptr, dev_ptr + size, dev_ptr, thrust::bit_xor<unsigned char>());

    return output;
}

unsigned char* _LaunchDNAEncoding(
    unsigned char* input, unsigned char* keyStream, unsigned char* output, int size) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    _DNAEncodingKernel<<<gridSize, blockSize>>>(input, keyStream, output, size);
    return output;
}

unsigned char* _LaunchPerformDNAOperation(
    unsigned char* input, unsigned char* keyStream, unsigned char* output, int size) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    _performDNAOperationKernel<<<gridSize, blockSize>>>(input, keyStream, output, size);
    return output;
}

//=========================================================================================================================




decryptionEngine::decryptionEngine(const imageData& inputImageDataLayout, const chenInitialArguments& inputChenArguments, const lorenzInitialArguments& inputLorenzArguments)
    : m_inputImageDataLayout{inputImageDataLayout}
    , m_chenInitialArguments{inputChenArguments}
    , m_lorenzInitialArguments{inputLorenzArguments}
{
    d_permutationMapping = nullptr; d_scratch_A = nullptr; d_scratch_B = nullptr;
    d_scratch_C = nullptr; d_scratch_D = nullptr; d_chaoticMask = nullptr;

    m_chenChaoticStreamRaw =    _chen3DChaoticStreamGeneration();
    m_lorenzChaoticStreamRaw =  _lorenz4DHyperChaoticStreamGeneration();

    int size = m_inputImageDataLayout.sizeOfImageFileInByte;
    _allocateDeviceScratchPadData(size);

    cudaMalloc((void**)&d_chaoticMask, size);
    cudaMemset(d_chaoticMask, 0x00, size);

    unsigned char* diffMask = _LaunchPixelDiffuse(d_chaoticMask, m_lorenzChaoticStreamRaw, d_scratch_A, size);
    unsigned char* dnaMask  = _LaunchDNAEncoding(diffMask, m_chenChaoticStreamRaw, d_scratch_B, size);
    
    // The final chaotic mask parks safely in d_chaoticMask forever!
    _LaunchPerformDNAOperation(dnaMask, m_lorenzChaoticStreamRaw, d_chaoticMask, size);
}
encryptionEngine::~encryptionEngine() {
    if (d_scratchA) { 
        cudaFree(d_scratchA); cudaFree(d_scratchB); 
        cudaFree(d_scratchC); cudaFree(d_scratchD); 
    }
    if (d_permMap)              cudaFree(d_permMap);
    if (m_chaoticStreamChen)    cudaFree(m_chaoticStreamChen);
    if (m_chaoticStreamLorenz)  cudaFree(m_chaoticStreamLorenz);
}
//this will return false on faliur
bool decryptionEngine::pushIntoTheQueue(imageData& inputImage){
    m_decryptionQueue.push(inputImage);
}
void decryptionEngine::run(){
    while(isRunning){

    }
}
void decryptionEngine::stop(){
    isRunning = false;
}
void decryptionEngine::pause(){
    isPaused = true;
}
void decryptionEngine::resume(){
    isPaused = false;
}

__global__ void _reversePermutationKernel(unsigned char* input , int* permutationMap , unsigned char* output , int size){
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index < size){
        output[index] = input[permutationMap[index]];
    }
}
unsigned char* decryptionEngine::_reversePermute(unsigned char* input,int* permutationMap,unsigned char* output , int size){
    int blockSize{256};
    int gridSize = (size + blockSize - 1) / blockSize;

    _reversePermutationKernel<<<gridSize , blockSize>>>(input , permutationMap , output , size);
    return output;
}
__global__ void _reverseDiffusionKernel(unsigned char* input, unsigned char* diffusionKey, unsigned char* output, int size) {
    long long index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        unsigned char current_cipher = input[index];
        // The first byte has no previous byte, so it XORs against 0!
        unsigned char prev_cipher = (index == 0) ? 0 : input[index - 1]; 
        unsigned char key = diffusionKey[index];
        
        output[index] = current_cipher ^ prev_cipher ^ key;
    }
}
unsigned char* decryptionEngine::_reverseDiffuse(unsigned char* input, unsigned char* diffusionKey , unsigned char* output , int size){
    int blockSize{256};
    int gridSize = (size + blockSize - 1) / blockSize;

    _reverseDiffusionKernel<<<gridSize , blockSize>>>(input , diffusionKey , output , size);
    return output;
}
__constant__ int d_DNA_ENCODING_RULES[8][4] = {
    {0, 1, 2, 3}, {0, 2, 1, 3}, {1, 0, 3, 2}, {1, 3, 0, 2},
    {3, 1, 2, 0}, {3, 2, 1, 0}, {2, 0, 3, 1}, {2, 3, 0, 1}
};
__device__ int decodeDNA(int encoded_base, int rule_index) {
    for (int i = 0; i < 4; ++i) {
        if (d_DNA_ENCODING_RULES[rule_index][i] == encoded_base) return i;
    }
    return 0; 
}
__global__ void _reverseDNAEncodingKernel(unsigned char* input, unsigned char* ruleKey, unsigned char* output, int size) {
    long long index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < size) {
        unsigned char cipher_byte = input[index];
        int rule = ruleKey[index] % 8; 

        unsigned char plain_byte = 0;
        for (int b = 0; b < 4; ++b) {
            // Extract the 2-bit biological base
            int encoded_base = (cipher_byte >> (2 * b)) & 3; 
            // Lookup its original index
            int decoded_base = decodeDNA(encoded_base, rule);
            // Pack it back into the plaintext byte
            plain_byte |= (decoded_base << (2 * b));
        }
        output[index] = plain_byte;
    }
}
unsigned char* decryptionEngine::_reverseDNAEncoding(unsigned char* input, unsigned char* ruleKey , unsigned char* output , int size){
    int blockSize{256};
    int gridSize = (size + blockSize - 1) / blockSize;

    _reverseDNAEncodingKernel<<<gridSize , blockSize>>>(input , ruleKey , output , size);
    return output;
}
__device__ int performDNA_Subtraction(int cipher_dna, int key_dna) {
    return (cipher_dna - key_dna + 4) % 4;
}
__global__ void _reverseDNAOperationKernel(unsigned char* input, unsigned char* chaoticStream, unsigned char* output, int size) {
    long long index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < size) {
        unsigned char cipher_byte = input[index];
        unsigned char key_byte = chaoticStream[index];

        unsigned char result_byte = 0;
        for (int b = 0; b < 4; ++b) {
            int cipher_base = (cipher_byte >> (2 * b)) & 3;
            int key_base = (key_byte >> (2 * b)) & 3;
            
            int subtracted_base = performDNA_Subtraction(cipher_base, key_base);
            result_byte |= (subtracted_base << (2 * b));
        }
        output[index] = result_byte;
    }
}
unsigned char* decryptionEngine::_reverseDNAOperation(unsigned char* input , unsigned char* chaoticStream , unsigned char* output , int size){
    int blockSize{256};
    int gridSize = (size + blockSize - 1) / blockSize;

    _reverseDNAOperationKernel<<<gridSize , blockSize>>>(input , chaoticStream , output , size);
    return output;
}
__global__ void _reverseImageZippingKernel(unsigned char* cipherText, unsigned char* branchChaoticPart, unsigned char* output, int size) {
    long long index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        // Ciphertext ^ LSB_Branch = MSB_Branch (The true payload)
        output[index] = cipherText[index] ^ branchChaoticPart[index];
    }
}
unsigned char* decryptionEngine::_reverseImageZipping(unsigned char* cipherText , unsigned char* branchChaoticPart , unsigned char* output, int size){
    int blockSize{256};
    int gridSize = (size + blockSize - 1) / blockSize;

    _reverseImageZippingKernel<<<gridSize , blockSize>>>(cipherText , branchChaoticPart , output , size);
    return output;
}

unsigned char* decryptionEngine::_decrypt(unsigned char* cipherTextImage, int size) {
    
    // 1. Load the incoming ciphertext frame into VRAM
    cudaMemcpy(d_scratch_B, cipherTextImage, size, cudaMemcpyHostToDevice);

    unsigned char* payload = _reverseImageZipping(d_scratch_B, d_chaoticMask, d_scratch_A, size);
    // Stage 6 Inverse: DNA Decoding this does nothing for now

    unsigned char* subDNA = _reverseDNAOperation(payload, m_lorenzChaoticStreamRaw, d_scratch_B, size);

    unsigned char* decDNA = _reverseDNAEncoding(subDNA, m_chenChaoticStreamRaw, d_scratch_A, size);

    unsigned char* unDiffuse = _reverseDiffuse(decDNA, m_lorenzChaoticStreamRaw, d_scratch_B, size);

    unsigned char* unPermute = _reversePermute(unDiffuse, d_permutationMapping, d_scratch_A, size);


    unsigned char* h_decrypted_output = new unsigned char[size];
    cudaMemcpy(h_decrypted_output, unPermute, size, cudaMemcpyDeviceToHost);

    return h_decrypted_output;
}