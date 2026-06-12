#include <iostream>
#include <filesystem>
#include <cuda_runtime.h> // The core CUDA API

#ifdef _WIN32
#include <windows.h>
extern "C" {
    __declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001;
}
#endif

#include "../vendor/stb/stb_image.h"
#include "../vendor/stb/stb_image_write.h"

namespace fs = std::filesystem;

// ---------------------------------------------------------
// THE GPU KERNEL (Executes on the Device)
// ---------------------------------------------------------
__global__ void grayscaleKernel(unsigned char* d_in, unsigned char* d_out, int width, int height, int channels) {
    // 1. Calculate the global thread ID (which pixel this specific thread is working on)
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // 2. Boundary check: Make sure we don't read outside the image array
    if (x < width && y < height) {
        // Map 2D coordinates back to the 1D contiguous memory index
        int gray_index = y * width + x; 
        int rgb_index = gray_index * channels;

        unsigned char r = d_in[rgb_index];
        unsigned char g = d_in[rgb_index + 1];
        unsigned char b = d_in[rgb_index + 2];

        // Apply the luminance formula
        d_out[gray_index] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
    }
}

// ---------------------------------------------------------
// THE CPU CONTROL FLOW (Executes on the Host)
// ---------------------------------------------------------
int main() {
    int width, height, original_channels;
    int desired_channels = 3; 
    
    // 1. Host Allocation (CPU RAM)
    //this function allocates the data on the system RAM and returns the starting address
    unsigned char *h_img = stbi_load("assets/input.png", &width, &height, &original_channels, desired_channels);
    if (!h_img) {
        printf("Error loading image.\n");
        return 1;
    }
    
    //this is the size of the total number of individual values 
    //or elements that are to be inside the sequential RAM
    size_t rgb_size             = width * height * desired_channels;    //here desired channel is the number of channels data that is to be loaded form the image file into the system ram as a contigous memory
    size_t gray_size            = width * height * 1;
    unsigned char *h_gray_img   = new unsigned char[gray_size];

    // 2. Device Allocation (GPU VRAM)
    unsigned char *d_in = nullptr;
    unsigned char *d_out = nullptr;
    cudaMalloc((void**)&d_in, rgb_size);        //allocates in the vram the mrmory for the data requried form the original image
    cudaMalloc((void**)&d_out, gray_size);      //allocates in the vram the memory for the gray scale image

    // 3. Transfer Host to Device (H2D)
    //      (the pointer to the vram allocated , pointer to image in the ram , vram allocated memory size , parameter flags)
    cudaMemcpy(d_in, h_img, rgb_size, cudaMemcpyHostToDevice);

    // 4. Kernel Execution
    // Define the grid size: blocks of 16x16 threads
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    
    // Launch the kernel!
    grayscaleKernel<<<gridSize, blockSize>>>(d_in, d_out, width, height, desired_channels);
    // Force CPU to wait until the GPU finishes processing
    cudaDeviceSynchronize();

    // 5. Transfer Device to Host (D2H)
    cudaMemcpy(h_gray_img, d_out, gray_size, cudaMemcpyDeviceToHost);


    //file saving part , save the file that is in the RAM into the disk
    std::string output_dir = "outputs";
    if (!fs::exists(output_dir)) fs::create_directory(output_dir);
    stbi_write_png((output_dir + "/gpu_grayscale.png").c_str(), width, height, 1, h_gray_img, width);

    // Cleanup Host and Device Memory
    cudaFree(d_in);
    cudaFree(d_out);
    stbi_image_free(h_img);
    delete[] h_gray_img;

    printf("GPU Processing Complete!\n");
    return 0;
}