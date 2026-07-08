#include"kernelsEncrypt.cuh"

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

__global__ void _pixelDiffusionXORMasking(unsigned char* input, unsigned char* diffusionMatrix, unsigned char* output, int size) {
    long long index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < size) {
        output[index] = input[index] ^ diffusionMatrix[index];
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

__global__ void _diffuseColumnTopToBottomKernel(unsigned char* data, unsigned char* chaoticStream, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= width) return;

    // Seed the initial value for the cascade using the first row's chaotic stream
    unsigned char prev_cipher = chaoticStream[col]; 

    for (int row = 0; row < height; ++row) {
        int idx = row * width + col;
        unsigned char p = data[idx];
        unsigned char k = chaoticStream[idx];

        // ARX Pipeline: Rotate -> Add -> XOR
        unsigned char mixed_prev    = (prev_cipher >> 3) | (prev_cipher << 5); 
        unsigned char sum           = p + mixed_prev; 
        unsigned char c             = sum ^ k;          

        data[idx]   = c;
        prev_cipher = c; // Feed forward
    }
}

__global__ void _diffuseColumnBottomToTopKernel(unsigned char* data, unsigned char* chaoticStream, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= width) return;

    unsigned char prev_cipher = chaoticStream[(height - 1) * width + col];

    for (int row = height - 1; row >= 0; --row) {
        int idx             = row * width + col;
        unsigned char p     = data[idx];
        unsigned char k     = chaoticStream[idx];

        unsigned char mixed_prev    = (prev_cipher >> 3) | (prev_cipher << 5);
        unsigned char sum           = p + mixed_prev;
        unsigned char c             = sum ^ k;

        data[idx] = c;
        prev_cipher = c;
    }
}

__global__ void _diffuseRowLeftToRightKernel(unsigned char* data, unsigned char* chaoticStream, int width, int height) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height) return;

    unsigned char prev_cipher = chaoticStream[row * width];

    for (int col = 0; col < width; ++col) {
        int idx = row * width + col;
        unsigned char p = data[idx];
        unsigned char k = chaoticStream[idx];

        unsigned char mixed_prev = (prev_cipher >> 3) | (prev_cipher << 5);
        unsigned char sum = p + mixed_prev;
        unsigned char c = sum ^ k;

        data[idx] = c;
        prev_cipher = c;
    }
}

__global__ void _diffuseRowRightToLeftKernel(unsigned char* data, unsigned char* chaoticStream, int width, int height) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height) return;

    unsigned char prev_cipher = chaoticStream[row * width + (width - 1)];

    for (int col = width - 1; col >= 0; --col) {
        int idx = row * width + col;
        unsigned char p = data[idx];
        unsigned char k = chaoticStream[idx];

        unsigned char mixed_prev = (prev_cipher >> 3) | (prev_cipher << 5);
        unsigned char sum = p + mixed_prev;
        unsigned char c = sum ^ k;

        data[idx] = c;
        prev_cipher = c;
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