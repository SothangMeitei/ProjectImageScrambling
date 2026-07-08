#include "kernelsDecrypt.cuh"
#include "sharedConstants.cuh"

__global__ void _reversePermutationKernel(unsigned char* input , int* permutationMap , unsigned char* output , int size){
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index < size){
        output[index] = input[permutationMap[index]];
    }
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


// ==============================================================================
// INVERSE PASS 1: ROW RIGHT-TO-LEFT
// ==============================================================================
__global__ void _inverseRowRightToLeftKernel(unsigned char* data, unsigned char* chaoticStream, int width, int height) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height) return;

    unsigned char prev_cipher = chaoticStream[row * width + (width - 1)];

    // Note: We traverse in the EXACT SAME direction as encryption 
    // because we must follow the dependency chain of the ciphertext.
    for (int col = width - 1; col >= 0; --col) {
        int idx = row * width + col;
        unsigned char c = data[idx];  // Read Ciphertext
        unsigned char k = chaoticStream[idx];

        // INVERSE ARX Pipeline
        unsigned char mixed_prev = (prev_cipher >> 3) | (prev_cipher << 5);
        unsigned char sum = c ^ k;          // 1. Reverse XOR
        unsigned char p = sum - mixed_prev; // 2. Reverse Addition (Modular Subtraction)

        data[idx] = p;     // Write Plaintext
        prev_cipher = c;   // 3. Feed forward the CIPHERTEXT (not plaintext)
    }
}

// ==============================================================================
// INVERSE PASS 2: ROW LEFT-TO-RIGHT
// ==============================================================================
__global__ void _inverseRowLeftToRightKernel(unsigned char* data, unsigned char* chaoticStream, int width, int height) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height) return;

    unsigned char prev_cipher = chaoticStream[row * width];

    for (int col = 0; col < width; ++col) {
        int idx = row * width + col;
        unsigned char c = data[idx];
        unsigned char k = chaoticStream[idx];

        unsigned char mixed_prev = (prev_cipher >> 3) | (prev_cipher << 5);
        unsigned char sum = c ^ k;
        unsigned char p = sum - mixed_prev;

        data[idx] = p;
        prev_cipher = c;
    }
}

// ==============================================================================
// INVERSE PASS 3: COLUMN BOTTOM-TO-TOP
// ==============================================================================
__global__ void _inverseColumnBottomToTopKernel(unsigned char* data, unsigned char* chaoticStream, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= width) return;

    unsigned char prev_cipher = chaoticStream[(height - 1) * width + col];

    for (int row = height - 1; row >= 0; --row) {
        int idx = row * width + col;
        unsigned char c = data[idx];
        unsigned char k = chaoticStream[idx];

        unsigned char mixed_prev = (prev_cipher >> 3) | (prev_cipher << 5);
        unsigned char sum = c ^ k;
        unsigned char p = sum - mixed_prev;

        data[idx] = p;
        prev_cipher = c;
    }
}

// ==============================================================================
// INVERSE PASS 4: COLUMN TOP-TO-BOTTOM
// ==============================================================================
__global__ void _inverseColumnTopToBottomKernel(unsigned char* data, unsigned char* chaoticStream, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= width) return;

    unsigned char prev_cipher = chaoticStream[col];

    for (int row = 0; row < height; ++row) {
        int idx = row * width + col;
        unsigned char c = data[idx];
        unsigned char k = chaoticStream[idx];

        unsigned char mixed_prev = (prev_cipher >> 3) | (prev_cipher << 5);
        unsigned char sum = c ^ k;
        unsigned char p = sum - mixed_prev;

        data[idx] = p;
        prev_cipher = c;
    }
}




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
__global__ void _reverseImageZippingKernel(unsigned char* cipherText, unsigned char* branchChaoticPart, unsigned char* output, int size) {
    long long index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        // Ciphertext ^ LSB_Branch = MSB_Branch (The true payload)
        output[index] = cipherText[index] ^ branchChaoticPart[index];
    }
}