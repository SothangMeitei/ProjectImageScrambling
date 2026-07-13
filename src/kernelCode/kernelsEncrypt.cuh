#pragma once
#include <cuda_runtime.h>
#include "sharedConstants.cuh"

__global__ void _bitReplaceKernel(unsigned char* input, unsigned char* output1, unsigned char* output2, int size);

__global__ void _pixelPermuteKernel(unsigned char* inputImage, unsigned char* outputImage, int* mapping, int size);

__global__ void _diffuseColumnTopToBottomKernel_Encrypt(unsigned char* input, unsigned char* diffusionKeyStream, int width , int height);

// __device__ char _diffusionModularChainingOperation(char previous , char current , char key);
// __device__ char _pixelMix(char byteValue ,char mixingKey);
// __global__ void _pixelDiffuseKernelByteMixing(unsigned char* input, unsigned char* diffusionKeyStream , unsigned char* output , int size);

// // --- New ARX Bi-Directional Diffusion Kernels ---
// __global__ void _diffuseColumnTopToBottomKernel(unsigned char* data, unsigned char* chaoticStream, int width, int height);
// __global__ void _diffuseColumnBottomToTopKernel(unsigned char* data, unsigned char* chaoticStream, int width, int height);
// __global__ void _diffuseRowLeftToRightKernel(unsigned char* data, unsigned char* chaoticStream, int width, int height);
// __global__ void _diffuseRowRightToLeftKernel(unsigned char* data, unsigned char* chaoticStream, int width, int height);

__device__ int encodeDNA(int binary_val, int rule_index);

__global__ void _DNAEncodingKernel(unsigned char* input, unsigned char* mapping, unsigned char* output, int size);

__device__ int performDNA_Addition(int pixel_dna, int key_dna);

__global__ void _performDNADecodingKernel(unsigned char* input, unsigned char* output, int size);

__global__ void _mergeTwoHalvesKernel(unsigned char* input1, unsigned char* input2, unsigned char* output, int size);