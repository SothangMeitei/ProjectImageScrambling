#pragma once
#include "encryptionEngine.h"

#include "../../vendor/stb/stb_image.h"
#include "../../vendor/stb/stb_image_write.h"
#include <filesystem>

#include<cstring>
#include"../chaoticStreamProcessing/chenStreamProcessor.h"
#include"../chaoticStreamProcessing/lorenzStreamProcessor.h"

#include<thread>

#include <cuda_runtime.h>
#include "../kernelCode/kernelsEncrypt.cuh"

namespace fs = std::filesystem;
static int g_streamWidth    = 640;
static int g_streamHeight   = 360;
static int g_streamChannels = 3;

//chaotic stream generation and then host to device transfer of data of the chaotic stream

unsigned char* encryptionEngine::chen3DChaoticStream() {
    int size = m_streamSize;

    chenChaoticSystem chenSolver(m_chenArguments, size);
    chenSolver.generate();
    chaoticStreamChen rawChen = chenSolver.getChaoticStreams();

    chenStreamProcessor permProcessor(size);
    permProcessor.ingestRawStream(rawChen.x);
    permProcessor.sortAndExtractMapping();

    cudaMalloc((void**)&d_permMap, size * sizeof(int));
    cudaMemcpy(d_permMap, permProcessor.getGPUFlatMapping(), size * sizeof(int), cudaMemcpyHostToDevice);

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
unsigned char* encryptionEngine::lorenz4DHyperChaoticStream() {
    size_t totalPayloadBytes = m_referenceFormat.sizeOfImageFileInByte;
    size_t requiredIntWords = (totalPayloadBytes + 3) / 4;

    lorenzChaoticSystem lorenzSolver(m_lorenzArguments, requiredIntWords);
    lorenzSolver.generate();

    lorenzStreamProcessor mint(requiredIntWords);
    mint.ingestRawStream(lorenzSolver.getChaoticStream().x);
    uint32_t* cpuIntReservoir = mint.getDiffusionValues();

    unsigned char* d_vramByteStream = nullptr;
    cudaMalloc((void**)&d_vramByteStream, totalPayloadBytes);
    cudaMemcpy(d_vramByteStream, cpuIntReservoir, totalPayloadBytes, cudaMemcpyHostToDevice);

    return d_vramByteStream; 
}

//engine initialization

encryptionEngine::encryptionEngine(
      const imageData& imageFormat 
    , const chenInitialArguments& chenArguments
    , const lorenzInitialArguments& lorenzArguments
    , const std::string& outputDir)
    
    : m_chenArguments{chenArguments} 
    , m_lorenzArguments{lorenzArguments}
    , m_streamSize{imageFormat.sizeOfImageFileInByte}
    , m_outputDir{outputDir}
{
    isRunning = true;
    isPaused  = false;
    m_referenceFormat = imageFormat;

    d_scratchA = nullptr; d_scratchB = nullptr; 
    d_scratchC = nullptr; d_scratchD = nullptr;
    m_currentArenaPixelSize = 0;

    d_permMap             = nullptr;
    m_chaoticStreamChen   = nullptr;
    m_chaoticStreamLorenz = nullptr;

    m_chaoticStreamChen   = chen3DChaoticStream();
    m_chaoticStreamLorenz = lorenz4DHyperChaoticStream();
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

void encryptionEngine::_reallocateVRAMScratchpadIfNeeded(size_t required_size) {
    if (required_size > m_currentArenaPixelSize) {
        if(d_scratchA) { 
            cudaFree(d_scratchA); cudaFree(d_scratchB); 
            cudaFree(d_scratchC); cudaFree(d_scratchD); 
        }
        cudaMalloc((void**)&d_scratchA, required_size);
        cudaMalloc((void**)&d_scratchB, required_size);
        cudaMalloc((void**)&d_scratchC, required_size);
        cudaMalloc((void**)&d_scratchD, required_size);
        m_currentArenaPixelSize = required_size;
    }
}

//internal functions operating on the data stored in the gpu and the cpu

std::pair<unsigned char*, unsigned char*> encryptionEngine::_LaunchBitReplace(
    unsigned char* inputImage, unsigned char* output1, unsigned char* output2, int size) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize; 
    _bitReplaceKernel<<<gridSize, blockSize>>>(inputImage, output1, output2, size);
    return {output1, output2};
}

unsigned char* encryptionEngine::_LaunchPixelPermute(
    unsigned char* inputImage, unsigned char* output, int* permutation, int size) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    _pixelPermuteKernel<<<gridSize, blockSize>>>(inputImage, output, permutation, size);
    return output;
}

unsigned char* encryptionEngine::_launchBiDirectionalARXDiffusion(unsigned char* d_data, unsigned char* d_chaoticStream, int width, int height) {
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

unsigned char* encryptionEngine::_LaunchDNAEncoding(
    unsigned char* input, unsigned char* keyStream, unsigned char* output, int size) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    _DNAEncodingKernel<<<gridSize, blockSize>>>(input, keyStream, output, size);
    return output;
}

unsigned char* encryptionEngine::_LaunchPerformDNAOperation(
    unsigned char* input, unsigned char* keyStream, unsigned char* output, int size) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    _performDNAOperationKernel<<<gridSize, blockSize>>>(input, keyStream, output, size);
    return output;
}

unsigned char* encryptionEngine::_LaunchDNADecoding(
    unsigned char* input, unsigned char* output, int size) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    _performDNADecodingKernel<<<gridSize, blockSize>>>(input, output, size);
    return output;
}

unsigned char* encryptionEngine::_LauchImageMerginZip(
    unsigned char* input1, unsigned char* input2, unsigned char* output, int size) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    _mergeTwoHalvesKernel<<<gridSize, blockSize>>>(input1, input2, output, size);
    return output;
}



unsigned char* encryptionEngine::_encrypt(unsigned char* plainTextInputImage, int size) {
    
    _reallocateVRAMScratchpadIfNeeded(size);
    cudaMemcpy(d_scratchB, plainTextInputImage, size, cudaMemcpyHostToDevice);

    std::pair<unsigned char*, unsigned char*> split = 
        _LaunchBitReplace(d_scratchB, d_scratchA, d_scratchC, size);

    unsigned char* permMSB = _LaunchPixelPermute(split.first,  d_scratchB, d_permMap, size);
    unsigned char* permLSB = _LaunchPixelPermute(split.second, d_scratchD, d_permMap, size);    //this too is useless ,as all 0's

    unsigned char* diffMSB = _launchBiDirectionalARXDiffusion(permMSB , m_chaoticStreamLorenz ,m_referenceFormat.width * m_referenceFormat.channels, m_referenceFormat.height);
    unsigned char* diffLSB = _launchBiDirectionalARXDiffusion(permLSB , m_chaoticStreamChen   ,m_referenceFormat.width * m_referenceFormat.channels, m_referenceFormat.height);

    unsigned char* dnaMSB = _LaunchDNAEncoding(diffMSB, m_chaoticStreamChen,   d_scratchA, size);
    unsigned char* dnaLSB = _LaunchDNAEncoding(diffLSB, m_chaoticStreamLorenz, d_scratchC, size);

    unsigned char* opMSB = _LaunchPerformDNAOperation(dnaMSB, m_chaoticStreamLorenz, d_scratchB, size);
    unsigned char* opLSB = _LaunchPerformDNAOperation(dnaLSB, m_chaoticStreamChen,   d_scratchD, size);

    unsigned char* decMSB = _LaunchDNADecoding(opMSB, d_scratchA, size);    //for now this is useless
    unsigned char* decLSB = _LaunchDNADecoding(opLSB, d_scratchC, size);    //this also the same case

    unsigned char* finalEncryptedDevice = _LauchImageMerginZip(decMSB, decLSB, d_scratchA, size);

    unsigned char* h_encrypted_output = new unsigned char[size];
    cudaMemcpy(h_encrypted_output, finalEncryptedDevice, size, cudaMemcpyDeviceToHost);

    return h_encrypted_output;
}

bool encryptionEngine::pushImageIntoQueueBuffer(const unsigned char* input) {
    imageData newFrame = m_referenceFormat; 
    newFrame.imagePixelValues = const_cast<unsigned char*>(input); 
    
    std::lock_guard<std::mutex> lock(m_queueMutex);
    m_inputImageQueueBuffer.push(newFrame);
    
    return true;
}

void encryptionEngine::stop() {
    isRunning = false;
}
void encryptionEngine::run() {
    int frame_count = 0;
    while(isRunning) {
        if(m_inputImageQueueBuffer.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2)); // Prevents 100% CPU core spin
            continue;
        }
        unsigned char* rawPixels{nullptr};
        int streamByteSize{0};


        { // CREATE A SCOPE BLOCK FOR THE MUTEX
            std::lock_guard<std::mutex> lock(m_queueMutex);
            
            if(m_inputImageQueueBuffer.empty()) {
                // If empty, the lock_guard is automatically released here
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                continue;
            }
            
            // Extract data and safely pop INSIDE the lock
            rawPixels = m_inputImageQueueBuffer.front().imagePixelValues;
            streamByteSize = m_inputImageQueueBuffer.front().sizeOfImageFileInByte;
            m_inputImageQueueBuffer.pop(); 
            
        }

        unsigned char* cipherText = _encrypt(rawPixels, streamByteSize);
        if (!fs::exists(m_outputDir)) fs::create_directory(m_outputDir);

        std::string out_path = m_outputDir + "/encrypted_frame_" + std::to_string(frame_count++) + ".png";
        stbi_write_png(out_path.c_str(), g_streamWidth, g_streamHeight, g_streamChannels, cipherText, g_streamWidth * g_streamChannels);

        delete[] cipherText;
    }    
}