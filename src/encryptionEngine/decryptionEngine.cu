
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




//================================================================================================================================================


unsigned char* _launchBiDirectionalARXDiffusion(unsigned char* d_data, unsigned char* d_chaoticStream, int width, int height) {
    int threadsPerBlock = 256;

    // 1. Calculate grid for Columns
    int blocksCol = (width + threadsPerBlock - 1) / threadsPerBlock;
    _diffuseColumnTopToBottomKernel<<<blocksCol, threadsPerBlock>>>(d_data, d_chaoticStream, width, height);
    cudaDeviceSynchronize();

    _diffuseColumnBottomToTopKernel<<<blocksCol, threadsPerBlock>>>(d_data, d_chaoticStream, width, height);
    cudaDeviceSynchronize();

    // 2. Calculate grid for Rows
    int blocksRow = (height + threadsPerBlock - 1) / threadsPerBlock;
    _diffuseRowLeftToRightKernel<<<blocksRow, threadsPerBlock>>>(d_data, d_chaoticStream, width, height);
    cudaDeviceSynchronize();

    _diffuseRowRightToLeftKernel<<<blocksRow, threadsPerBlock>>>(d_data, d_chaoticStream, width, height);
    cudaDeviceSynchronize();

    return d_data;
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

decryptionEngine::decryptionEngine(
    const imageData& inputImageDataLayout
    , const chenInitialArguments& inputChenArguments
    , const lorenzInitialArguments& inputLorenzArguments
    , const std::string& outputDir)

    : m_inputImageDataLayout{inputImageDataLayout}
    , m_chenInitialArguments{inputChenArguments}
    , m_lorenzInitialArguments{inputLorenzArguments}
    , m_outputDir{outputDir}
{
    d_permutationMapping = nullptr; d_scratch_A = nullptr; d_scratch_B = nullptr;
    d_scratch_C = nullptr; d_scratch_D = nullptr; d_chaoticMask = nullptr;

    m_chenChaoticStreamRaw =    _chen3DChaoticStreamGeneration();
    m_lorenzChaoticStreamRaw =  _lorenz4DHyperChaoticStreamGeneration();

    int size = m_inputImageDataLayout.sizeOfImageFileInByte;
    _allocateDeviceScratchPadData(size);

    cudaMalloc((void**)&d_chaoticMask, size);
    cudaMemset(d_chaoticMask, 0x00, size);

    unsigned char* diffMask = _launchBiDirectionalARXDiffusion(d_chaoticMask, m_chenChaoticStreamRaw, m_inputImageDataLayout.width, m_inputImageDataLayout.height);
    unsigned char* dnaMask  = _LaunchDNAEncoding(diffMask, m_lorenzChaoticStreamRaw, d_scratch_B, size);

    _LaunchPerformDNAOperation(dnaMask, m_chenChaoticStreamRaw, d_chaoticMask, size);
}
decryptionEngine::~decryptionEngine() {
    if (d_scratch_A) { 
        cudaFree(d_scratch_A); cudaFree(d_scratch_B); 
        cudaFree(d_scratch_C); cudaFree(d_scratch_D); 
    }
    if (d_permutationMapping)               cudaFree(d_permutationMapping);
    if (d_chaoticMask)                      cudaFree(d_chaoticMask);
    if (m_chenChaoticStreamRaw)             cudaFree(m_chenChaoticStreamRaw);
    if (m_lorenzChaoticStreamRaw)           cudaFree(m_lorenzChaoticStreamRaw);
}
//this will return false on failur
bool decryptionEngine::pushIntoTheQueue(imageData& inputImage){
    std::lock_guard<std::mutex> lock(m_queueMutex);
    m_decryptionQueue.push(inputImage);
    return true;
}
void decryptionEngine::run() {
    int frame_count = 0;
    while(isRunning) {
        if(m_decryptionQueue.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            continue;
        }
        
        unsigned char* cipherPixels{nullptr};
        int streamByteSize{0};

        { // CREATE A SCOPE BLOCK FOR THE MUTEX
            std::lock_guard<std::mutex> lock(m_queueMutex);
            
            if(m_decryptionQueue.empty()) {
                // If empty, the lock_guard is automatically released here
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                continue;
            }
            
            // Extract data and safely pop INSIDE the lock
            cipherPixels = m_decryptionQueue.front().imagePixelValues;
            streamByteSize = m_decryptionQueue.front().sizeOfImageFileInByte;
            m_decryptionQueue.pop();
        }
        unsigned char* plainText = _decrypt(cipherPixels, streamByteSize);

        if(!fs::exists(m_outputDir)) fs::create_directory(m_outputDir);

        std::string out_path = m_outputDir + "/frames_" + std::to_string(frame_count++) + ".png";
        
        stbi_write_png(out_path.c_str(), 
                       m_inputImageDataLayout.width, 
                       m_inputImageDataLayout.height, 
                       m_inputImageDataLayout.channels, 
                       plainText, 
                       m_inputImageDataLayout.width * m_inputImageDataLayout.channels);

        delete[] plainText;
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

unsigned char* decryptionEngine::_reversePermute(unsigned char* input,int* permutationMap,unsigned char* output , int size){
    int blockSize{256};
    int gridSize = (size + blockSize - 1) / blockSize;

    _reversePermutationKernel<<<gridSize , blockSize>>>(input , permutationMap , output , size);
    return output;
}

unsigned char* decryptionEngine::_reverseDiffuse(unsigned char* input, unsigned char* diffusionKey , unsigned char* output , int size){
    int blockSize{256};
    int gridSize = (size + blockSize - 1) / blockSize;

    _reverseDiffusionKernel<<<gridSize , blockSize>>>(input , diffusionKey , output , size);
    return output;
}

unsigned char* decryptionEngine::_launchBiDirectionalARXInverseDiffusion(unsigned char* d_data, unsigned char* d_chaoticStream, int width, int height) {
    int threadsPerBlock = 256;

    // 1. Calculate grid for Rows (Executed FIRST during Decryption)
    int blocksRow = (height + threadsPerBlock - 1) / threadsPerBlock;
    _inverseRowRightToLeftKernel<<<blocksRow, threadsPerBlock>>>(d_data, d_chaoticStream, width, height);
    cudaDeviceSynchronize();

    _inverseRowLeftToRightKernel<<<blocksRow, threadsPerBlock>>>(d_data, d_chaoticStream, width, height);
    cudaDeviceSynchronize();

    // 2. Calculate grid for Columns (Executed LAST during Decryption)
    int blocksCol = (width + threadsPerBlock - 1) / threadsPerBlock;
    _inverseColumnBottomToTopKernel<<<blocksCol, threadsPerBlock>>>(d_data, d_chaoticStream, width, height);
    cudaDeviceSynchronize();

    _inverseColumnTopToBottomKernel<<<blocksCol, threadsPerBlock>>>(d_data, d_chaoticStream, width, height);
    cudaDeviceSynchronize();

    return d_data;
}

unsigned char* decryptionEngine::_reverseDNAEncoding(unsigned char* input, unsigned char* ruleKey , unsigned char* output , int size){
    int blockSize{256};
    int gridSize = (size + blockSize - 1) / blockSize;

    _reverseDNAEncodingKernel<<<gridSize , blockSize>>>(input , ruleKey , output , size);
    return output;
}

unsigned char* decryptionEngine::_reverseDNAOperation(unsigned char* input , unsigned char* chaoticStream , unsigned char* output , int size){
    int blockSize{256};
    int gridSize = (size + blockSize - 1) / blockSize;

    _reverseDNAOperationKernel<<<gridSize , blockSize>>>(input , chaoticStream , output , size);
    return output;
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

    unsigned char* unDiffuse = _launchBiDirectionalARXInverseDiffusion(
        decDNA
        , m_lorenzChaoticStreamRaw
        , m_inputImageDataLayout.width * m_inputImageDataLayout.channels
        , m_inputImageDataLayout.height);

    unsigned char* unPermute = _reversePermute(unDiffuse, d_permutationMapping, d_scratch_B, size);

    unsigned char* h_decrypted_output = new unsigned char[size];
    cudaMemcpy(h_decrypted_output, unPermute, size, cudaMemcpyDeviceToHost);

    return h_decrypted_output;
}