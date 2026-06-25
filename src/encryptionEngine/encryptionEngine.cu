#pragma once
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

namespace fs = std::filesystem;
static int g_streamWidth    = 640;
static int g_streamHeight   = 360;
static int g_streamChannels = 3;

// ============================================================================
// 1. THE GPU KERNELS
// ============================================================================

__global__ void _bitReplaceKernel(unsigned char* input, unsigned char* output1, unsigned char* output2, int size) {
    long long index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        output1[index] = input[index];
        output2[index] = 0x00; // Blank chaotic mask
    }
}

__global__ void _pixelPermuteKernel(unsigned char* inputImage, unsigned char* outputImage, int* mapping, int size) {
    long long index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < size) {
        outputImage[mapping[index]] = inputImage[index];
    }
}

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

__global__ void _performDNADecodingKernel(unsigned char* input, unsigned char* output, int size) {
    long long index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < size) {
        output[index] = input[index];
    }
}

__global__ void _mergeTwoHalvesKernel(unsigned char* input1, unsigned char* input2, unsigned char* output, int size) {
    long long index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        //this merges the two image files using the XOR that is does not have the bias like OR operation do
        output[index] = input1[index] ^ input2[index]; // Bitwise XOR removes all probabilistic bias
    }
}


// ============================================================================
// 2. CHAOTIC STREAM GENERATORS & VRAM BRIDGES
// ============================================================================

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

// ============================================================================
// 3. ENGINE LIFECYCLE
// ============================================================================

encryptionEngine::encryptionEngine(
      const imageData& imageFormat 
    , const chenInitialArguments& chenArguments
    , const lorenzInitialArguments& lorenzArguments)
    
    : m_chenArguments{chenArguments} 
    , m_lorenzArguments{lorenzArguments}
    , m_streamSize{imageFormat.sizeOfImageFileInByte}
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

unsigned char* encryptionEngine::_LaunchPixelDiffuse(
    unsigned char* inputImage, unsigned char* diffusionMatrix, unsigned char* output, int size) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    _pixelDiffuseKernel<<<gridSize, blockSize>>>(inputImage, diffusionMatrix, output, size);

    // O(log N) Inclusive Prefix XOR Scan natively in VRAM
    thrust::device_ptr<unsigned char> dev_ptr(output);
    thrust::inclusive_scan(dev_ptr, dev_ptr + size, dev_ptr, thrust::bit_xor<unsigned char>());

    return output;
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

// ============================================================================
// 4. THE MASTER RELAY EXECUTION
// ============================================================================

unsigned char* encryptionEngine::_encrypt(unsigned char* plainTextInputImage, int size) {
    
    _reallocateVRAMScratchpadIfNeeded(size);
    cudaMemcpy(d_scratchB, plainTextInputImage, size, cudaMemcpyHostToDevice);

    std::pair<unsigned char*, unsigned char*> split = 
        _LaunchBitReplace(d_scratchB, d_scratchA, d_scratchC, size);

    unsigned char* permMSB = _LaunchPixelPermute(split.first,  d_scratchB, d_permMap, size);
    unsigned char* permLSB = _LaunchPixelPermute(split.second, d_scratchD, d_permMap, size);    //this too is useless ,as all 0's

    // Cross-Pollinated Keys (Breaks correlation collisions)
    unsigned char* diffMSB = _LaunchPixelDiffuse(permMSB, m_chaoticStreamLorenz, d_scratchA, size);
    unsigned char* diffLSB = _LaunchPixelDiffuse(permLSB, m_chaoticStreamChen,   d_scratchC, size);

    unsigned char* dnaMSB = _LaunchDNAEncoding(diffMSB, m_chaoticStreamChen,   d_scratchB, size);
    unsigned char* dnaLSB = _LaunchDNAEncoding(diffLSB, m_chaoticStreamLorenz, d_scratchD, size);

    unsigned char* opMSB = _LaunchPerformDNAOperation(dnaMSB, m_chaoticStreamLorenz, d_scratchA, size);
    unsigned char* opLSB = _LaunchPerformDNAOperation(dnaLSB, m_chaoticStreamChen,   d_scratchC, size);

    unsigned char* decMSB = _LaunchDNADecoding(opMSB, d_scratchB, size);    //for now this is useless
    unsigned char* decLSB = _LaunchDNADecoding(opLSB, d_scratchD, size);    //this also the same case

    unsigned char* finalEncryptedDevice = _LauchImageMerginZip(decMSB, decLSB, d_scratchA, size);

    unsigned char* h_encrypted_output = new unsigned char[size];
    cudaMemcpy(h_encrypted_output, finalEncryptedDevice, size, cudaMemcpyDeviceToHost);

    return h_encrypted_output;
}

bool encryptionEngine::pushImageIntoQueueBuffer(const unsigned char* input) {
    imageData newFrame = m_referenceFormat; 
    newFrame.imagePixelValues = const_cast<unsigned char*>(input); 
    m_inputImageQueueBuffer.push(newFrame);
    return true;
}

void encryptionEngine::stop() {
    isRunning = false;
}

void encryptionEngine::run() {
    int frame_count = 0;
    while(isRunning) {
        if(m_inputImageQueueBuffer.empty()) continue;
        
        unsigned char* rawPixels = m_inputImageQueueBuffer.front().imagePixelValues;
        int streamByteSize       = m_inputImageQueueBuffer.front().sizeOfImageFileInByte;

        unsigned char* cipherText = _encrypt(rawPixels, streamByteSize);
        m_inputImageQueueBuffer.pop();

        std::string output_dir = "outputs";
        if (!fs::exists(output_dir)) fs::create_directory(output_dir);

        std::string out_path = output_dir + "/encrypted_frame_" + std::to_string(frame_count++) + ".png";
        stbi_write_png(out_path.c_str(), g_streamWidth, g_streamHeight, g_streamChannels, cipherText, g_streamWidth * g_streamChannels);

        delete[] cipherText;
    }    
}