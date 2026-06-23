#pragma once
#include <cuda_runtime.h>
#include "encryptionEngine.h"

#include "../vendor/stb/stb_image.h"
#include "../vendor/stb/stb_image_write.h"
#include <filesystem>

#include<cstring>
#include"../chaoticStreamProcessing/chenStreamProcessor.h"

namespace fs = std::filesystem;

// ============================================================================
// 1. THE GPU KERNELS
// ============================================================================

__global__ void _bitReplaceKernel(unsigned char* input, unsigned char* output1, unsigned char* output2, int size) {
    long long index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        unsigned char current_byte = input[index];    
        unsigned char buffer1 = 0;

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
        int rule = mapping[index] % 8; 

        // Safely extract 2-bit chunks to prevent out-of-bounds constant memory access
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
        // Identity copy: Preserves the Galois field ciphertext bits safely
        output[index] = input[index];
    }
}

__global__ void _mergeTwoHalvesKernel(unsigned char* input1, unsigned char* input2, unsigned char* output, int size) {
    long long index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        output[index] = input1[index] | input2[index];
    }
}

// ============================================================================
// 2. THE HOST DISPATCH WRAPPERS (Strictly matching .h signatures!)
// ============================================================================

// ============================================================================
// 1. CHAOTIC STREAM GENERATORS & VRAM BRIDGES (CPU Host Execution)
// ============================================================================

unsigned char* encryptionEngine::chen3DChaoticStream() {
    int size = m_streamSize;

    // 1. Fire up the Chen RK4 Differential Solver on CPU RAM
    chenChaoticSystem chenSolver(m_chenArguments, size);
    chenSolver.generate();
    chaoticStreamChen rawChen = chenSolver.getChaoticStreams();

    // 2. Build GPU Permutation Map (d_permMap) using Chen Axis X
    chenStreamProcessor permProcessor(size);
    permProcessor.ingestRawStream(rawChen.x);
    permProcessor.sortAndExtractMapping();

    // Bridge the flat permutation indices contiguously across PCIe to VRAM
    cudaMalloc((void**)&d_permMap, size * sizeof(int));
    cudaMemcpy(d_permMap, permProcessor.getGPUFlatMapping(), size * sizeof(int), cudaMemcpyHostToDevice);

    // 3. Build Watson-Crick Polymorphic Rule Stream using Chen Axis Y
    // Extract raw mantissa entropy and bound to legal rules (0 to 7)
    unsigned char* h_dnaRules = new unsigned char[size];
    for (int i = 0; i < size; ++i) {
        uint32_t bits;
        std::memcpy(&bits, &rawChen.y[i], sizeof(float));
        h_dnaRules[i] = (unsigned char)(bits & 7); 
    }

    unsigned char* d_dnaRulesDevice = nullptr;
    cudaMalloc((void**)&d_dnaRulesDevice, size);
    cudaMemcpy(d_dnaRulesDevice, h_dnaRules, size, cudaMemcpyHostToDevice);

    delete[] h_dnaRules;
    return d_dnaRulesDevice; // Assigned directly to chaoticStreamChen!
}

unsigned char* encryptionEngine::lorenz4DHyperChaoticStream() {
    int size = m_streamSize;

    // 1. Fire up the 4D Hyper-Lorenz RK4 Solver on CPU RAM
    lorenzChaoticSystem lorenzSolver(m_lorenzArguments, size);
    lorenzSolver.generate();
    chaoticStreamLorenz rawLorenz = lorenzSolver.getChaoticStream();

    // 2. Build Galois Addition & Diffusion Key Stream using Lorenz Axis X
    // Harvest contiguous 8-bit byte entropy directly out of the mantissas
    unsigned char* h_lorenzBytes = new unsigned char[size];
    for (int i = 0; i < size; ++i) {
        uint32_t bits;
        std::memcpy(&bits, &rawLorenz.x[i], sizeof(float));
        h_lorenzBytes[i] = (unsigned char)(bits & 0xFF); // Bounded 0-255 byte masking
    }

    unsigned char* d_lorenzStreamDevice = nullptr;
    cudaMalloc((void**)&d_lorenzStreamDevice, size);
    cudaMemcpy(d_lorenzStreamDevice, h_lorenzBytes, size, cudaMemcpyHostToDevice);

    delete[] h_lorenzBytes;
    return d_lorenzStreamDevice; // Assigned directly to chaoticStreamLorenz!
}

// ============================================================================
// 2. ENGINE LIFECYCLE (The Ignition Switch)
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

    d_scratchA = nullptr; d_scratchB = nullptr; 
    d_scratchC = nullptr; d_scratchD = nullptr;
    m_currentArenaPixelSize = 0;

    d_permMap           = nullptr;
    m_chaoticStreamChen   = nullptr;
    m_chaoticStreamLorenz = nullptr;

    std::cout << "[ENGINE BOOT]: Igniting Chen 3D & Lorenz 4D Differential Solvers...\n";
    
    // Generate CPU chaos, execute Radix sorting, and bridge pointers to VRAM
    m_chaoticStreamChen   = chen3DChaoticStream();
    m_chaoticStreamLorenz = lorenz4DHyperChaoticStream();

    std::cout << "[ENGINE BOOT]: Master Secret Key VRAM Arenas successfully populated.\n";
}

encryptionEngine::~encryptionEngine() {
    // Scrub all secret cryptographic buffers from physical silicon
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
// 3. THE MASTER RELAY EXECUTION
// ============================================================================

unsigned char* encryptionEngine::_encrypt(unsigned char* plainTextInputImage, int size) {
    
    _reallocateVRAMScratchpadIfNeeded(size);

    // Copy host plaintext image into VRAM landing pad (ScratchB)
    cudaMemcpy(d_scratchB, plainTextInputImage, size, cudaMemcpyHostToDevice);

    // --- THE POINTER RELAY (Capturing returns per .h contract) ---

    // Stage 1: Split ScratchB ---> returns {ScratchA, ScratchC}
    std::pair<unsigned char*, unsigned char*> split = 
        _LaunchBitReplace(d_scratchB, d_scratchA, d_scratchC, size);

    // Stage 2: Permute split halves ---> returns ScratchB and ScratchD
    unsigned char* permMSB = _LaunchPixelPermute(split.first,  d_scratchB, d_permMap, size);
    unsigned char* permLSB = _LaunchPixelPermute(split.second, d_scratchD, d_permMap, size);

    // Stage 3: Diffuse permuted halves ---> returns ScratchA and ScratchC
    unsigned char* diffMSB = _LaunchPixelDiffuse(permMSB, m_chaoticStreamLorenz, d_scratchA, size);
    unsigned char* diffLSB = _LaunchPixelDiffuse(permLSB, m_chaoticStreamLorenz, d_scratchC, size);

    // Stage 4: DNA Encode diffused halves ---> returns ScratchB and ScratchD
    unsigned char* dnaMSB = _LaunchDNAEncoding(diffMSB, m_chaoticStreamChen, d_scratchB, size);
    unsigned char* dnaLSB = _LaunchDNAEncoding(diffLSB, m_chaoticStreamChen, d_scratchD, size);

    // Stage 5: Perform DNA Addition ---> returns ScratchA and ScratchC
    unsigned char* opMSB = _LaunchPerformDNAOperation(dnaMSB, m_chaoticStreamLorenz, d_scratchA, size);
    unsigned char* opLSB = _LaunchPerformDNAOperation(dnaLSB, m_chaoticStreamLorenz, d_scratchC, size);

    // Stage 6: DNA Decode ---> returns ScratchB and ScratchD
    unsigned char* decMSB = _LaunchDNADecoding(opMSB, d_scratchB, size);
    unsigned char* decLSB = _LaunchDNADecoding(opLSB, d_scratchD, size);

    // Stage 7: Zip decoded halves contiguously back together ---> returns ScratchA
    unsigned char* finalEncryptedDevice = _LauchImageMerginZip(decMSB, decLSB, d_scratchA, size);

    // Harvest finished ciphertext back to Host RAM
    unsigned char* h_encrypted_output = new unsigned char[size];
    cudaMemcpy(h_encrypted_output, finalEncryptedDevice, size, cudaMemcpyDeviceToHost);

    return h_encrypted_output;
}

unsigned char* encryptionEngine::_decrypt(unsigned char* cypherTextImage, int size) {
    return nullptr; // Placeholder stub matching .h
}

bool encryptionEngine::pushImageIntoQueueBuffer(const unsigned char* input) {
    return true;    // Placeholder stub matching .h
}

void encryptionEngine::stop() {
    isRunning = false;
}

void encryptionEngine::run() {
    int frame_count = 0;
    while(isRunning) {
        if(m_inputImageQueueBuffer.empty()) continue;
        
        // Harvest spatial metadata from your queue struct
        int width    = m_inputImageQueueBuffer.front().width;
        int height   = m_inputImageQueueBuffer.front().height;
        int channels = m_inputImageQueueBuffer.front().channels;
        int size     = width * height * channels;

        unsigned char* cipherText = _encrypt(m_inputImageQueueBuffer.front().imagePixelValues, size);
        m_inputImageQueueBuffer.pop();

        // Ensure output directory exists contiguously on disk
        std::string output_dir = "outputs";
        if (!fs::exists(output_dir)) fs::create_directory(output_dir);

        // Write the encrypted byte stream to disk as a lossless PNG
        std::string out_path = output_dir + "/encrypted_frame_" + std::to_string(frame_count++) + ".png";
        stbi_write_png(out_path.c_str(), width, height, channels, cipherText, width * channels);

        // Safely free the harvested host RAM buffer
        delete[] cipherText;
    }    
}