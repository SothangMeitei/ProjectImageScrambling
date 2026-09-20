#include"kernelsEncrypt.cuh"

__global__ void _bitReplaceKernel(unsigned char* input, unsigned char* output1, unsigned char* output2, int size) {
    long long index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        unsigned char current_byte = input[index];
        unsigned char new_byte = 0;

        #pragma unroll  //this is for loop unrolling 
        for(int i = 0 ; i < 4 ; ++i){
            int b = (current_byte >> i) & 1;            
            int mapped_val = 2 - b; 
            new_byte |= (mapped_val << (i * 2));
        }
        output1[index] = new_byte;
        new_byte = 0;

        #pragma unroll
        for(int i = 0 ; i < 4 ; ++i){
            int b = (current_byte >> (i + 4)) & 1;
            int mapped_val = 2 - b;
            new_byte |= (mapped_val << (i * 2));
        }
        output2[index] = new_byte;
    }
}

__global__ void _pixelPermuteKernel(unsigned char* inputImage, unsigned char* outputImage, int* mapping, int size) {
    long long index = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(index < size) {
        long long target_idx = mapping[index];
        
        if (target_idx >= size || target_idx < 0) {
            target_idx = index; // Fallback to no-permutation for this specific pixel
        }
        
        outputImage[target_idx] = inputImage[index];
    }
}

/*
    Instead of XORing chaining we do chaning in the manner of taking the vlaue of the previous and then doing 
    some form of modular arithmatic that is non linear over the XOR operation , is also non assosiative 
    
    this combined with the mixing will make the delta propagate to all the pixels and also multiply the difference with
    guareeanted non losing of the differnece when this sequenced chainning is taking place    
*/

/*
    We need to somehow introduce the idea of the ARX 
    Add     : modular addition for the porpagation of the carray bit
    Rotate  : to solve the problem of the mod addition against the flipping of the MSB
    Xoring  : XOR with the chaotic stream , this introduces the confusion  
*/

__device__ __forceinline__ unsigned char rotl8(unsigned char val, int r) {
    return (val << r) | (val >> (8 - r));
}

__global__ void _diffuseColumnTopToBottomKernel_Encrypt(unsigned char* data, unsigned char* chaoticStream, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= width) return;

    unsigned char prev_cipher = chaoticStream[col]; 

    for (int row = 0; row < height; ++row) {
        long long idx = (long long)row * width + col;
        
        unsigned char plain_text = data[idx];
        unsigned char k = chaoticStream[idx];

        //why roatate right by just a fixed value of 3
        unsigned char cipher_text = rotl8((unsigned char)(plain_text + prev_cipher + k), 3) ^ k;

        data[idx] = cipher_text;
        prev_cipher = cipher_text; 
    }
}

__global__ void _diffuseColumnTopToBottomKernel(unsigned char* data, unsigned char* chaoticStream, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= width) return;

    unsigned char prev_cipher = chaoticStream[col]; 

    for (int row = 0; row < height; ++row) {
        long long idx = (long long)row * width + col;
        
        unsigned char plain_text = data[idx];
        unsigned char k = chaoticStream[idx];

        unsigned char cipher_text = rotl8((unsigned char)(plain_text + prev_cipher + k), 3) ^ k;

        data[idx] = cipher_text;
        prev_cipher = cipher_text; 
    }
}

__global__ void _diffuseRowLeftToRightKernel(unsigned char* data, unsigned char* chaoticStream, int width, int height) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height) return;

    unsigned char prev_cipher = chaoticStream[(long long)row * width];

    for (int col = 0; col < width; ++col) {
        long long idx = (long long)row * width + col;

        unsigned char plain_text = data[idx];
        unsigned char k = chaoticStream[idx];

        unsigned char cipher_text = rotl8((unsigned char)(plain_text + prev_cipher + k), 3) ^ k;

        data[idx] = cipher_text;
        prev_cipher = cipher_text;
    }
}

__global__ void _diffuseColumnBottomToTopKernel(unsigned char* data, unsigned char* chaoticStream, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= width) return;

    unsigned char prev_cipher = chaoticStream[(long long)(height - 1) * width + col];

    for (int row = height - 1; row >= 0; --row) {
        long long idx = (long long)row * width + col;

        unsigned char plain_text = data[idx];
        unsigned char k = chaoticStream[idx];

        unsigned char cipher_text = rotl8((unsigned char)(plain_text + prev_cipher + k), 3) ^ k;

        data[idx] = cipher_text;
        prev_cipher = cipher_text;
    }
}

__global__ void _diffuseRowRightToLeftKernel(unsigned char* data, unsigned char* chaoticStream, int width, int height) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height) return;

    unsigned char prev_cipher = chaoticStream[(long long)row * width + (width - 1)];

    for (int col = width - 1; col >= 0; --col) {
        long long idx = (long long)row * width + col;

        unsigned char plain_text = data[idx];
        unsigned char k = chaoticStream[idx];

        unsigned char cipher_text = rotl8((unsigned char)(plain_text + prev_cipher + k), 3) ^ k;

        data[idx] = cipher_text;
        prev_cipher = cipher_text;
    }
}

__constant__ int d_DNA_ENCODING_RULES[8][4] = {
    {0, 1, 2, 3}, {0, 2, 1, 3}, {1, 0, 3, 2}, {1, 3, 0, 2},
    {3, 1, 2, 0}, {3, 2, 1, 0}, {2, 0, 3, 1}, {2, 3, 0, 1}
};
//a is 0 , g is 1 , c is 2 , and t is 3
__constant__ int d_DNA_ADDITION_RULES[4][4] = {
    0 , 1 , 2 , 3 ,
    1 , 2 , 3 , 0 ,
    2 , 3 , 0 , 1 ,
    3 , 0 , 1 , 2 
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
            int base_val = (current_byte >> (2 * b)) & 3; //use bit masking to extract the two bits that represent the DNA base
            int encoded_base = encodeDNA(base_val, rule);
            buffer |= (encoded_base << (2 * b));
        }
        output[index] = buffer;
    }
}

__device__ int performDNA_Addition(int dna_neucleophile1, int dna_neucleophile2) {
    return d_DNA_ADDITION_RULES[dna_neucleophile1][dna_neucleophile2];
}
//merge the dna encoded image file
__global__ void _mergeTwoHalvesKernel(unsigned char* input1, unsigned char* input2, unsigned char* output, int size) {
    long long index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        unsigned char byte1 = input1[index];
        unsigned char byte2 = input2[index];
        unsigned char result_byte = 0;

        // Unpack the 8-bit byte into four 2-bit bases, add them, and repack
        for (int b = 0; b < 4; ++b) {
            int base1 = (byte1 >> (2 * b)) & 3;
            int base2 = (byte2 >> (2 * b)) & 3;
            
            int added_base = performDNA_Addition(base1, base2);
            result_byte |= (added_base << (2 * b));
        }
        
        output[index] = result_byte;
    }
}