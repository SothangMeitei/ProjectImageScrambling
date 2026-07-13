
#pragma once

#include"decryptionEngine.h"
#include <cuda_runtime.h>

#include "../../vendor/stb/stb_image.h"
#include "../../vendor/stb/stb_image_write.h"
#include <filesystem>

#include<cstring>
#include"../chaoticStreamProcessing/chenStreamProcessor.h"
#include"../chaoticStreamProcessing/lorenzStreamProcessor.h"

#include<thread>
#include <cerrno>
#include <cstring>

#include "../kernelCode/kernelsDecrypt.cuh"
#include "../kernelCode/kernelsEncrypt.cuh"

namespace fs = std::filesystem;

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

decryptionEngine::decryptionEngine(
    const imageData& inputImageDataLayout
    , const chenInitialArguments& inputChenArguments
    , const lorenzInitialArguments& inputLorenzArguments)

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
}
decryptionEngine::~decryptionEngine() {
    if(d_scratch_A) cudaFree(d_scratch_A);
    if(d_scratch_B) cudaFree(d_scratch_B);
    if(d_scratch_C) cudaFree(d_scratch_C);
    if(d_scratch_D) cudaFree(d_scratch_D);

    if (d_permutationMapping)               cudaFree(d_permutationMapping);
    if (d_chaoticMask)                      cudaFree(d_chaoticMask);
    if (m_chenChaoticStreamRaw)             cudaFree(m_chenChaoticStreamRaw);
    if (m_lorenzChaoticStreamRaw)           cudaFree(m_lorenzChaoticStreamRaw);
}

unsigned char* decryptionEngine::_reverseBitReplacement(unsigned char* input1 , unsigned char* input2 , unsigned char* output , int size){
    //take the two bytes of the inputs
    //and then when merging we take the input1's as the first bit and the input2 and the second bit
    //and then for first bit second bit = 10 -> 1 or 01 -> 0
    long long blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    _reverseBitReplacementKernel<<<gridSize , blockSize>>>(input1 , input2 , output , size);
    return output;
}
unsigned char* decryptionEngine::_reversePermute(unsigned char* input,int* permutationMap,unsigned char* output , int size){
    int blockSize{256};
    int gridSize = (size + blockSize - 1) / blockSize;

    _reversePermutationKernel<<<gridSize , blockSize>>>(input , permutationMap , output , size);
    return output;
}

unsigned char* decryptionEngine::_reverseDiffuse(unsigned char* d_data, unsigned char* d_chaoticStream, int width, int height) {
    int threadsPerBlock = 256;

    // CORRECTED: Calculate grid for Columns, not Rows!
    int blocksCol = (width + threadsPerBlock - 1) / threadsPerBlock;
    _reverseDiffusionKernel<<<blocksCol, threadsPerBlock>>>(d_data, d_chaoticStream, width, height);
    cudaDeviceSynchronize();

    return d_data;
}

unsigned char* decryptionEngine::_reverseDNAEncoding(unsigned char* input, unsigned char* ruleKey , unsigned char* output , int size){
    int blockSize{256};
    int gridSize = (size + blockSize - 1) / blockSize;

    _reverseDNAEncodingKernel<<<gridSize , blockSize>>>(input , ruleKey , output , size);
    return output;
}

unsigned char* decryptionEngine::_reverseImageZipping(unsigned char* cipherText , unsigned char* cipherText1 , unsigned char* output, int size){
    int blockSize{256};
    int gridSize = (size + blockSize - 1) / blockSize;

    _reverseDNAImageMergingKernel<<<gridSize , blockSize>>>(cipherText , cipherText1 , output , size);
    return output;
}

unsigned char* decryptionEngine::decrypt(unsigned char* cipherTextImage, unsigned char* cipherTextImage1, int size) {
    
    // --- THE TRIPWIRE ---
    auto check_cuda = [](const std::string& step) {
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            std::cerr << "\n======================================================\n";
            std::cerr << "[GPU CRASH CAUGHT EXACTLY AT]: " << step << "\n";
            std::cerr << "Error Desc: " << cudaGetErrorString(err) << "\n";
            std::cerr << "======================================================\n";
            exit(1);
        }
    };

    // 1. Load the incoming ciphertexts into VRAM
    // Main -> B  |  Aux -> D
    cudaMemcpy(d_scratch_B, cipherTextImage, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_scratch_D, cipherTextImage1, size, cudaMemcpyHostToDevice);
    check_cuda("Initial Memcpy Host->Device");

    // 2. REVERSE IMAGE ZIPPING (DNA Subtraction)
    // Output: Unzipped MSB payload goes to A. (D still holds the aux LSB payload).
    unsigned char* payloadMSB = _reverseImageZipping(d_scratch_B, d_scratch_D, d_scratch_A, size);
    check_cuda("Reverse Image Zipping Kernel");

    // 3. REVERSE DNA ENCODING
    // Input: A -> Output: B  |  Input: D -> Output: C
    unsigned char* decDNA_MSB = _reverseDNAEncoding(payloadMSB, m_chenChaoticStreamRaw, d_scratch_B, size);
    check_cuda("Reverse DNA Encoding Kernel (MSB)");

    unsigned char* decDNA_LSB = _reverseDNAEncoding(d_scratch_D, m_lorenzChaoticStreamRaw, d_scratch_C, size);
    check_cuda("Reverse DNA Encoding Kernel (LSB)");

    // 4. REVERSE DIFFUSION (In-Place)
    // Input/Output: B  |  Input/Output: C
    int w_ch = m_inputImageDataLayout.width * m_inputImageDataLayout.channels;
    int h = m_inputImageDataLayout.height;
    
    unsigned char* unDiffuseMSB = _reverseDiffuse(decDNA_MSB, m_lorenzChaoticStreamRaw, w_ch, h);
    check_cuda("Reverse Diffusion Kernel (MSB)");

    unsigned char* unDiffuseLSB = _reverseDiffuse(decDNA_LSB, m_chenChaoticStreamRaw, w_ch, h);
    check_cuda("Reverse Diffusion Kernel (LSB)");

    // 5. REVERSE PERMUTATION
    // Input: B -> Output: A  |  Input: C -> Output: D
    unsigned char* unPermuteMSB = _reversePermute(unDiffuseMSB, d_permutationMapping, d_scratch_A, size);
    check_cuda("Reverse Permute Kernel (MSB)");

    unsigned char* unPermuteLSB = _reversePermute(unDiffuseLSB, d_permutationMapping, d_scratch_D, size);
    check_cuda("Reverse Permute Kernel (LSB)");

    // 6. REVERSE BIT REPLACEMENT (Combine the 2 halves)
    // Input: A + D -> Output: B (The final un-split plaintext!)
    unsigned char* reversed_bit_replacement_output = _reverseBitReplacement(unPermuteMSB, unPermuteLSB, d_scratch_B, size);
    check_cuda("Reverse Bit Replacement Kernel");

    // ====================================================================
    // EXPORT TO CPU
    // ====================================================================
    unsigned char* h_plain_text = new unsigned char[size];
    cudaMemcpy(h_plain_text, reversed_bit_replacement_output, size, cudaMemcpyDeviceToHost);
    
    return h_plain_text;
}