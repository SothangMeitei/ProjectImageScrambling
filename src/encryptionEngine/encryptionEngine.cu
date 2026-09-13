#include "encryptionEngine.h"
#include "../../vendor/stb/stb_image.h"
#include "../../vendor/stb/stb_image_write.h"
#include <filesystem>
#include <cstring>
#include <cmath>
#include "../chaoticStreamProcessing/chenStreamProcessor.h"
#include "../chaoticStreamProcessing/lorenzStreamProcessor.h"
#include <thread>
#include <cerrno>
#include <cuda_runtime.h>
#include "../kernelCode/kernelsEncrypt.cuh"
#include <fstream>
#include <memory>
#include <stdexcept>

namespace fs = std::filesystem;

unsigned char* encryptionEngine::chen3DChaoticStream() { // (Same for decryptionEngine)
    int size = m_streamSize;
    chenChaoticSystem<double> chenSolver(m_chenArguments, size);
    chenSolver.generate();
    chaoticStreamChen<double> rawChen = chenSolver.getChaoticStreams();
    
    chenStreamProcessor permProcessor(size);
    permProcessor.ingestRawStream(rawChen); // Ingests full struct!
    permProcessor.sortAndExtractMapping();
    
    cudaMalloc((void**)&d_permMap, size * sizeof(int));
    cudaMemcpy(d_permMap, permProcessor.getGPUFlatMapping(), size * sizeof(int), cudaMemcpyHostToDevice);

    // Directly copy the processed byte stream from the processor
    unsigned char* d_dnaRulesDevice = nullptr;
    cudaMalloc((void**)&d_dnaRulesDevice, size);
    cudaMemcpy(d_dnaRulesDevice, permProcessor.getByteStream(), size, cudaMemcpyHostToDevice);
    
    return d_dnaRulesDevice;
}

unsigned char* encryptionEngine::lorenz4DHyperChaoticStream() {
    size_t totalPayloadBytes = m_referenceFormat.sizeOfImageFileInByte;
    size_t requiredIntWords  = (totalPayloadBytes + 3) / 4;

    lorenzChaoticSystem<double> lorenzSolver(m_lorenzArguments, requiredIntWords);
    lorenzSolver.generate();

    lorenzStreamProcessor mint(requiredIntWords);
    mint.ingestRawStream(lorenzSolver.getChaoticStream());

    uint32_t* cpuIntReservoir = mint.getDiffusionValues();
    unsigned char* d_vramByteStream = nullptr;

    cudaMalloc((void**)&d_vramByteStream, totalPayloadBytes);
    cudaMemcpy(d_vramByteStream, cpuIntReservoir, totalPayloadBytes, cudaMemcpyHostToDevice);

    return d_vramByteStream;
}

encryptionEngine::encryptionEngine(
      const imageData& imageFormat
    , const chenInitialArguments<double>& chenArguments
    , const lorenzInitialArguments<double>& lorenzArguments)
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
    if (d_scratchA) cudaFree(d_scratchA);
    if (d_scratchB) cudaFree(d_scratchB);
    if (d_scratchC) cudaFree(d_scratchC);
    if (d_scratchD) cudaFree(d_scratchD);
    if (d_permMap)             cudaFree(d_permMap);
    if (m_chaoticStreamChen)   cudaFree(m_chaoticStreamChen);
    if (m_chaoticStreamLorenz) cudaFree(m_chaoticStreamLorenz);
}

void encryptionEngine::_reallocateVRAMScratchpadIfNeeded(size_t required_size) {
    if (required_size > m_currentArenaPixelSize) {
        if (d_scratchA) cudaFree(d_scratchA);
        if (d_scratchB) cudaFree(d_scratchB);
        if (d_scratchC) cudaFree(d_scratchC);
        if (d_scratchD) cudaFree(d_scratchD);

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
    cudaDeviceSynchronize();
    return {output1, output2};
}

unsigned char* encryptionEngine::_LaunchPixelPermute(unsigned char* input, unsigned char* output, int* mapping, int size) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    _pixelPermuteKernel<<<gridSize, blockSize>>>(input, output, mapping, size);
    cudaDeviceSynchronize();
    return output;
}

unsigned char* encryptionEngine::_LaunchPixelDiffusion(unsigned char* d_data, unsigned char* d_chaoticStream, int width, int height) {
    int threadsPerBlock = 256;
    int blocksCol = (width + threadsPerBlock - 1) / threadsPerBlock;
    _diffuseColumnTopToBottomKernel_Encrypt<<<blocksCol, threadsPerBlock>>>(d_data, d_chaoticStream, width, height);
    cudaDeviceSynchronize();
    return d_data;
}

unsigned char* encryptionEngine::_LaunchPixelDiffusion2D(unsigned char* d_data, unsigned char* d_chaoticStream, int width, int height) {
    int threadsPerBlock = 256;
    int blocksCol = (width + threadsPerBlock - 1) / threadsPerBlock;
    int blocksRow = (height + threadsPerBlock - 1) / threadsPerBlock;

    // Pass 1: Column Top-to-Bottom
    _diffuseColumnTopToBottomKernel<<<blocksCol, threadsPerBlock>>>(d_data, d_chaoticStream, width, height);
    cudaDeviceSynchronize();

    // Pass 2: Row Left-to-Right
    _diffuseRowLeftToRightKernel<<<blocksRow, threadsPerBlock>>>(d_data, d_chaoticStream, width, height);
    cudaDeviceSynchronize();

    // Pass 3: Column Bottom-to-Top
    _diffuseColumnBottomToTopKernel<<<blocksCol, threadsPerBlock>>>(d_data, d_chaoticStream, width, height);
    cudaDeviceSynchronize();

    // Pass 4: Row Right-to-Left
    _diffuseRowRightToLeftKernel<<<blocksRow, threadsPerBlock>>>(d_data, d_chaoticStream, width, height);
    cudaDeviceSynchronize();

    return d_data;
}

unsigned char* encryptionEngine::_LaunchDNAEncoding(unsigned char* input, unsigned char* keyStream, unsigned char* output, int size) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    _DNAEncodingKernel<<<gridSize, blockSize>>>(input, keyStream, output, size);
    cudaDeviceSynchronize();
    return output;
}

unsigned char* encryptionEngine::_LauchImageMerginZip(unsigned char* input1, unsigned char* input2, unsigned char* output, int size) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    _mergeTwoHalvesKernel<<<gridSize, blockSize>>>(input1, input2, output, size);
    cudaDeviceSynchronize();
    return output;
}

void encryptionEngine::exportKeystreams(const std::string& outputDirectory, int size) {
    static bool already_exported = false;
    if (already_exported) return;

    try {
        std::string chenPath   = outputDirectory + "/keystream_chen.bin";
        std::string lorenzPath = outputDirectory + "/keystream_lorenz.bin";
        std::unique_ptr<unsigned char[]> h_buffer(new unsigned char[size]);

        cudaError_t errChen = cudaMemcpy(h_buffer.get(), m_chaoticStreamChen, size, cudaMemcpyDeviceToHost);
        if (errChen == cudaSuccess) {
            std::ofstream outChen(chenPath, std::ios::binary);
            outChen.write(reinterpret_cast<char*>(h_buffer.get()), size);
            outChen.close();
        }

        cudaError_t errLorenz = cudaMemcpy(h_buffer.get(), m_chaoticStreamLorenz, size, cudaMemcpyDeviceToHost);
        if (errLorenz == cudaSuccess) {
            std::ofstream outLorenz(lorenzPath, std::ios::binary);
            outLorenz.write(reinterpret_cast<char*>(h_buffer.get()), size);
            outLorenz.close();
        }

        std::cout << "      [SYS METRICS] : Processed Keystreams safely exported for NIST analysis." << std::endl;
        already_exported = true;
    } catch (const std::exception& e) {
        std::cerr << "  [FATAL ERROR IN EXPORT]: " << e.what() << std::endl;
    }
}

std::pair<unsigned char*, unsigned char*> encryptionEngine::encrypt(unsigned char* plainTextInputImage, int size) {
    auto check_cuda = [](const std::string& step) {
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            std::string errStr = std::string("\n[GPU CRASH AT]: ") + step + " | " + cudaGetErrorString(err) + "\n";
            throw std::runtime_error(errStr);
        }
    };

    _reallocateVRAMScratchpadIfNeeded(size);

    cudaMemcpy(d_scratchB, plainTextInputImage, size, cudaMemcpyHostToDevice);

    std::pair<unsigned char*, unsigned char*> split = _LaunchBitReplace(d_scratchB, d_scratchA, d_scratchC, size);
    check_cuda("Bit Replacement Kernel");

    unsigned char* permMSB = _LaunchPixelPermute(split.first,  d_scratchB, d_permMap, size);
    check_cuda("Pixel Permute Kernel (MSB)");

    unsigned char* permLSB = _LaunchPixelPermute(split.second, d_scratchD, d_permMap, size);
    check_cuda("Pixel Permute Kernel (LSB)");

    int w_ch = m_referenceFormat.width * m_referenceFormat.channels;
    int h    = m_referenceFormat.height;

    unsigned char* diffMSB = _LaunchPixelDiffusion2D(permMSB, m_chaoticStreamLorenz, w_ch, h);
    check_cuda("Pixel Diffusion 2D Kernel (MSB)");

    unsigned char* diffLSB = _LaunchPixelDiffusion2D(permLSB, m_chaoticStreamChen, w_ch, h);
    check_cuda("Pixel Diffusion 2D Kernel (LSB)");

    unsigned char* dnaMSB = _LaunchDNAEncoding(diffMSB, m_chaoticStreamChen, d_scratchA, size);
    check_cuda("DNA Encoding Kernel (MSB)");

    unsigned char* dnaLSB = _LaunchDNAEncoding(diffLSB, m_chaoticStreamLorenz, d_scratchC, size);
    check_cuda("DNA Encoding Kernel (LSB)");

    unsigned char* finalEncryptedDevice = _LauchImageMerginZip(dnaMSB, dnaLSB, d_scratchB, size);
    check_cuda("Image Merging Zip Kernel");

    unsigned char* h_main_cipher = new unsigned char[size];
    unsigned char* h_aux_cipher  = new unsigned char[size];

    cudaMemcpy(h_main_cipher, finalEncryptedDevice, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_aux_cipher, dnaLSB, size, cudaMemcpyDeviceToHost);

    return {h_main_cipher, h_aux_cipher};
}