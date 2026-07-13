#include "kernelsDecrypt.cuh"
#include "sharedConstants.cuh"

__global__ void _reverseBitReplacementKernel(unsigned char* input1, unsigned char* input2, unsigned char* output, int size) {
    long long index = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (index < size) {
        unsigned char byte1 = input1[index];
        unsigned char byte2 = input2[index];
        unsigned char recovered_byte = 0;

        #pragma unroll
        for(int i = 0 ; i < 4 ; ++i){
            int b = (byte1 >> (i * 2)) & 1; 
            recovered_byte |= (b << i);
        }
        #pragma unroll
        for(int i = 0 ; i < 4 ; ++i){
            int b = (byte2 >> (i * 2)) & 1; 
            recovered_byte |= (b << (i + 4));
        }
        output[index] = recovered_byte;
    }
}
__global__ void _reversePermutationKernel(unsigned char* input , int* permutationMap , unsigned char* output , int size){
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index < size){
        output[index] = input[permutationMap[index]];
    }
}
__global__ void _reverseDiffusionKernel(unsigned char* data, unsigned char* chaoticStream, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= width) return;

    // The exact same seed used in encryption
    unsigned char prev_cipher = chaoticStream[col]; 

    for (int row = 0; row < height; ++row) {
        long long idx = row * width + col;
        
        unsigned char cipher_text = data[idx];
        unsigned char k = chaoticStream[idx];

        unsigned char plain_text = cipher_text ^ prev_cipher ^ k;

        data[idx] = plain_text;
        prev_cipher = cipher_text; 
    }
}

//a is 0 , g is 1 , c is 2 , and t is 3
__constant__ int d_DNA_SUBSTRACTION_RULES[4][4] = {
    0 , 3 , 2 , 1 ,
    1 , 0 , 3 , 2 ,
    2 , 1 , 0 , 3 ,
    3 , 2 , 1 , 0 
};
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
__device__ int performDNA_Subtraction(int dna_neucleotide1, int dna_neucleotide2) {
    return d_DNA_SUBSTRACTION_RULES[dna_neucleotide1][dna_neucleotide2];
}
__global__ void _reverseDNAImageMergingKernel(unsigned char* cipherText, unsigned char* cipherDNA1, unsigned char* output, int size) {
    long long index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        unsigned char byte1 = cipherText[index];
        unsigned char byte2 = cipherDNA1[index];
        unsigned char result_byte = 0;

        // Unpack the 8-bit byte into four 2-bit bases, subtract them, and repack
        for (int b = 0; b < 4; ++b) {
            int base1 = (byte1 >> (2 * b)) & 3;
            int base2 = (byte2 >> (2 * b)) & 3;
            
            int sub_base = performDNA_Subtraction(base1, base2);
            result_byte |= (sub_base << (2 * b));
        }
        
        output[index] = result_byte;
    }
}