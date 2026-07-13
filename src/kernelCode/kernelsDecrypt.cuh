#pragma once
__global__ void _reverseBitReplacementKernel(unsigned char* input1 , unsigned char* input2 , unsigned char* output , int size);
__global__ void _reversePermutationKernel(unsigned char* input , int* permutationMap , unsigned char* output , int size);
__global__ void _reverseDiffusionKernel(unsigned char* input, unsigned char* diffusionKey, int width, int height);

// // --- Inverse ARX Bi-Directional Diffusion Kernels ---
// __global__ void _inverseRowRightToLeftKernel(unsigned char* data, unsigned char* chaoticStream, int width, int height);
// __global__ void _inverseRowLeftToRightKernel(unsigned char* data, unsigned char* chaoticStream, int width, int height);
// __global__ void _inverseColumnBottomToTopKernel(unsigned char* data, unsigned char* chaoticStream, int width, int height);
// __global__ void _inverseColumnTopToBottomKernel(unsigned char* data, unsigned char* chaoticStream, int width, int height);

__device__ int decodeDNA(int encoded_base, int rule_index);
__global__ void _reverseDNAEncodingKernel(unsigned char* input, unsigned char* ruleKey, unsigned char* output, int size);
__device__ int performDNA_Subtraction(int cipher_dna, int key_dna);
__global__ void _reverseDNAImageMergingKernel(unsigned char* cipherText, unsigned char* branchChaoticPart, unsigned char* output, int size);