#pragma once
#include "encryptionEngine.h"

#include "../../vendor/stb/stb_image.h"
#include "../../vendor/stb/stb_image_write.h"
#include <filesystem>

#include<cstring>
#include"../chaoticStreamProcessing/chenStreamProcessor.h"
#include"../chaoticStreamProcessing/lorenzStreamProcessor.h"

#include<thread>
#include <cerrno>
#include <cstring>

#include <cuda_runtime.h>
#include "../kernelCode/kernelsEncrypt.cuh"

namespace fs = std::filesystem;

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
    , const lorenzInitialArguments& lorenzArguments)
    
    : m_chenArguments{chenArguments} 
    , m_lorenzArguments{lorenzArguments}
    , m_streamSize{imageFormat.sizeOfImageFileInByte}
{
    m_referenceFormat = imageFormat;
    d_scratchA = nullptr; d_scratchB = nullptr; 
    d_scratchC = nullptr; d_scratchD = nullptr;
    m_currentArenaPixelSize = 0;

    d_permMap             = nullptr;
    m_chaoticStreamChen   = chen3DChaoticStream();
    m_chaoticStreamLorenz = lorenz4DHyperChaoticStream();
}

encryptionEngine::~encryptionEngine() {
    if(d_scratchA) cudaFree(d_scratchA);
    if(d_scratchB) cudaFree(d_scratchB);
    if(d_scratchC) cudaFree(d_scratchC);
    if(d_scratchD) cudaFree(d_scratchD);

    if (d_permMap)              cudaFree(d_permMap);
    if (m_chaoticStreamChen)    cudaFree(m_chaoticStreamChen);
    if (m_chaoticStreamLorenz)  cudaFree(m_chaoticStreamLorenz);
}

void encryptionEngine::_reallocateVRAMScratchpadIfNeeded(size_t required_size) {
    if (required_size > m_currentArenaPixelSize) {
        if(d_scratchA) cudaFree(d_scratchA);
        if(d_scratchB) cudaFree(d_scratchB);
        if(d_scratchC) cudaFree(d_scratchC);
        if(d_scratchD) cudaFree(d_scratchD);
        
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
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "[KERNEL CRASH]: _LaunchBitReplace failed -> " << cudaGetErrorString(err) << "\n";
    } else {
        std::cout << "return from the bit replacement\n";
    }

    return {output1, output2};
}

unsigned char* encryptionEngine::_LaunchPixelPermute(unsigned char* input, unsigned char* output, int* mapping, int size) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    
    _pixelPermuteKernel<<<gridSize, blockSize>>>(input, output, mapping, size);
    
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "\n[KERNEL CRASH]: _LaunchPixelPermute failed -> " << cudaGetErrorString(err) << "\n";
    } else {
        std::cout << "return from the pixel permute\n";
    }
    
    return output;
}
unsigned char* encryptionEngine::_LaunchPixelDiffusion(unsigned char* d_data, unsigned char* d_chaoticStream, int width, int height) {
    int threadsPerBlock = 256;

    // 1. Calculate grid for Columns
    int blocksCol = (width + threadsPerBlock - 1) / threadsPerBlock;
    _diffuseColumnTopToBottomKernel_Encrypt<<<blocksCol, threadsPerBlock>>>(d_data, d_chaoticStream, width, height);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "[KERNEL CRASH]: _LaunchPixelDiffusion failed -> " << cudaGetErrorString(err) << "\n";
    } else {
        std::cout << "return from the diffusion function\n";
    }
    return d_data;
}

unsigned char* encryptionEngine::_LaunchDNAEncoding(
    unsigned char* input, unsigned char* keyStream, unsigned char* output, int size) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    _DNAEncodingKernel<<<gridSize, blockSize>>>(input, keyStream, output, size);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "[KERNEL CRASH]: _LaunchDNAEncoding failed -> " << cudaGetErrorString(err) << "\n";
    } else {
        std::cout << "return from the dna encoding function\n";
    }
    return output;
}

unsigned char* encryptionEngine::_LauchImageMerginZip(
    unsigned char* input1, unsigned char* input2, unsigned char* output, int size) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    _mergeTwoHalvesKernel<<<gridSize, blockSize>>>(input1, input2, output, size);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "[KERNEL CRASH]: _LauchImageMerginZip failed -> " << cudaGetErrorString(err) << "\n";
    } else {
        std::cout << "return from the merge function\n";
    }
    return output;
}

std::pair<unsigned char*, unsigned char*> encryptionEngine::encrypt(unsigned char* plainTextInputImage, int size) {
    
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

    _reallocateVRAMScratchpadIfNeeded(size);
    cudaMemcpy(d_scratchB, plainTextInputImage, size, cudaMemcpyHostToDevice);
    check_cuda("Initial Memcpy Host->Device");

    // 1. RUN BIT REPLACEMENT (Split Image)
    // Input: B  |  Output MSB -> A  |  Output LSB -> C
    std::pair<unsigned char*, unsigned char*> split = _LaunchBitReplace(d_scratchB, d_scratchA, d_scratchC, size);
    check_cuda("Bit Replacement Kernel");

    // 2. RUN PIXEL PERMUTE
    // Input: A -> Output: B  |  Input: C -> Output: D
    unsigned char* permMSB = _LaunchPixelPermute(split.first,  d_scratchB, d_permMap, size);
    check_cuda("Pixel Permute Kernel (MSB)");

    unsigned char* permLSB = _LaunchPixelPermute(split.second, d_scratchD, d_permMap, size);
    check_cuda("Pixel Permute Kernel (LSB)");

    // 3. RUN PIXEL DIFFUSION (In-Place)
    // Input/Output: B  |  Input/Output: D
    int w_ch = m_referenceFormat.width * m_referenceFormat.channels;
    int h = m_referenceFormat.height;
    
    unsigned char* diffMSB = _LaunchPixelDiffusion(permMSB, m_chaoticStreamLorenz, w_ch, h);
    check_cuda("Pixel Diffusion Kernel (MSB)");

    unsigned char* diffLSB = _LaunchPixelDiffusion(permLSB, m_chaoticStreamChen, w_ch, h);
    check_cuda("Pixel Diffusion Kernel (LSB)");

    // 4. RUN DNA ENCODING
    // Input: B -> Output: A  |  Input: D -> Output: C
    unsigned char* dnaMSB = _LaunchDNAEncoding(diffMSB, m_chaoticStreamChen, d_scratchA, size);
    check_cuda("DNA Encoding Kernel (MSB)");

    unsigned char* dnaLSB = _LaunchDNAEncoding(diffLSB, m_chaoticStreamLorenz, d_scratchC, size);
    check_cuda("DNA Encoding Kernel (LSB)");

    // 5. RUN IMAGE ZIPPING (DNA Addition)
    // Input: A + C  |  Output: B (The final merged ciphertext)
    unsigned char* finalEncryptedDevice = _LauchImageMerginZip(dnaMSB, dnaLSB, d_scratchB, size);
    check_cuda("Image Merging Zip Kernel");

    // ====================================================================
    // EXPORT TO CPU
    // ====================================================================
    unsigned char* h_main_cipher = new unsigned char[size];
    unsigned char* h_aux_cipher  = new unsigned char[size];

    cudaMemcpy(h_main_cipher, finalEncryptedDevice, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_aux_cipher, dnaLSB, size, cudaMemcpyDeviceToHost); // The un-zipped LSB branch

    return {h_main_cipher, h_aux_cipher};
}