#include <iostream>
#include <filesystem>
#include <vector>
#include <string>
#include <thread>
#include <chrono>

// Instruct STB vendor headers to compile their native C bytecode implementation inline
#define STB_IMAGE_IMPLEMENTATION
#include "../vendor/stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../vendor/stb/stb_image_write.h"

// Include your overarching encryption engine and chaotic parameter contracts
#include "encryptionEngine/encryptionEngine.h"
#include "chaoticSystems/chenChaoticSystem.h"
#include "chaoticSystems/lorenzHyperChaoticSystem.h"

namespace fs = std::filesystem;

int main() {
    std::cout << "============================================================\n";
    std::cout << "    HIGH-THROUGHPUT DNA CHAOTIC CRYPTO ENGINE (RTX 2050)    \n";
    std::cout << "============================================================\n\n";

    // ------------------------------------------------------------------------
    // STAGE 1: I/O CONTAINER SANITIZATION & FILE GATHERING
    // ------------------------------------------------------------------------
    std::string asset_dir  = "assets";
    std::string output_dir = "outputs";

    if (!fs::exists(asset_dir)) {
        std::cerr << "[SYSTEM ERROR]: Asset container directory '" << asset_dir << "' not found.\n";
        return 1;
    }
    if (!fs::exists(output_dir)) {
        fs::create_directory(output_dir);
        std::cout << "[HOST I/O]: Created contiguous output destination '" << output_dir << "/'\n";
    }

    // Harvest all image file paths (PNG/JPG) contiguously from the assets directory
    std::vector<std::string> image_filepaths;
    for (const auto& entry : fs::directory_iterator(asset_dir)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            // Convert extension to lowercase for robust matching
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp") {
                image_filepaths.push_back(entry.path().string());
            }
        }
    }

    if (image_filepaths.empty()) {
        std::cerr << "[SYSTEM ERROR]: No valid image assets found inside '" << asset_dir << "/'.\n";
        return 1;
    }

    std::cout << "[HOST I/O]: Successfully harvested " << image_filepaths.size() << " raw video frames contiguously.\n\n";

    // ------------------------------------------------------------------------
    // STAGE 2: THE IGNITION PEEK & MASTER KEY GENERATION
    // ------------------------------------------------------------------------
    std::cout << "[STAGE 2]: Decoding initial frame metadata to bound VRAM Arena...\n";
    
    int width, height, original_channels;
    int desired_channels = 3; // Enforcing contiguous 3-channel RGB byte packing
    
    // Peek at the first frame to capture the stream's absolute spatial dimensions
    unsigned char* h_firstFrameBytes = stbi_load(
        image_filepaths[0].c_str(), &width, &height, &original_channels, desired_channels
    );

    if (!h_firstFrameBytes) {
        std::cerr << "[SYSTEM ERROR]: Failed to decode initial image asset: " << image_filepaths[0] << "\n";
        return 1;
    }

    int total_stream_bytes = width * height * desired_channels;

    // Package the captured metadata into your explicit constructor contract
    imageData streamFormat;
    streamFormat.imagePixelValues      = h_firstFrameBytes;
    streamFormat.sizeOfImageFileInByte = total_stream_bytes;
    streamFormat.width                 = width;
    streamFormat.height                = height;
    streamFormat.channels              = desired_channels;

    // --- GENERATE MASTER SECRET KEYS (The unguessable initial coordinates) ---
    std::cout << "[STAGE 2]: Instantiating Master Secret Keys (Kerckhoffs-Compliant)...\n";
    
    // Chen: (a=35, b=3, c=28), 1000 discard cycles, starting coordinate X, Y, Z
    chenInitialArguments chenSecretKeys(
        35.0f, 3.0f, 28.0f, 1000, 
        0.1234567f, 0.5432198f, 0.9876543f
    );

    // Lorenz: (a=10, b=8/3, c=46, d=2, e=12), 1500 discard cycles, starting coord X, Y, Z, W
    lorenzInitialArguments lorenzSecretKeys(
        10.0f, (8.0f / 3.0f), 46.0f, 2.0f, 12.0f, 1500, 
        0.1111111f, 0.2222222f, 0.3333333f, 0.4444444f
    );

    // Boot the hardware: Ignites RK4 CPU solvers, executes Radix sort, allocates static VRAM arena!
    std::cout << "[SYSTEM BOOT]: Igniting master encryptionEngine hardware...\n";
    encryptionEngine masterEngine(streamFormat, chenSecretKeys, lorenzSecretKeys);

    // We must push the first frame bytes into the queue buffer since we loaded them!
    masterEngine.pushImageIntoQueueBuffer(h_firstFrameBytes);

    // ------------------------------------------------------------------------
    // STAGE 3: ASYNCHRONOUS MULTI-THREADED PIPELINE DISPATCH
    // ------------------------------------------------------------------------
    std::cout << "\n[STAGE 3]: Spawning dedicated background GPU Compute Worker Thread...\n";
    // Spawns a background thread running the consumer polling loop: masterEngine.run()
    std::thread computeWorker(&encryptionEngine::run, &masterEngine);

    // PRODUCER LOOP: Primary CPU thread streams the remaining files contiguously into RAM
    std::cout << "[PRODUCER]: Streaming remaining directory frames into engine queue...\n";
    for (size_t i = 1; i < image_filepaths.size(); ++i) {
        int w, h, c;
        unsigned char* h_frameBytes = stbi_load(
            image_filepaths[i].c_str(), &w, &h, &c, desired_channels
        );

        if (h_frameBytes) {
            // Push raw host pointer into thread-safe queue buffer contiguously
            masterEngine.pushImageIntoQueueBuffer(h_frameBytes);
            std::cout << "  -> Queued Frame [" << i << "/" << (image_filepaths.size()-1) << "] : " << image_filepaths[i] << "\n";
        } else {
            std::cerr << "[SYSTEM WARNING]: Dropped corrupted frame asset: " << image_filepaths[i] << "\n";
        }
    }

    // ------------------------------------------------------------------------
    // STAGE 4: PIPELINE DRAIN & GRACEFUL HARDWARE JOIN
    // ------------------------------------------------------------------------
    std::cout << "\n[PRODUCER]: Directory ingestion complete. Draining VRAM compute arena...\n";
    
    // Give the GPU worker thread a brief temporal window to finish popping the final frames,
    // executing the Galois math, and exporting the PNGs to disk.
    std::this_thread::sleep_for(std::chrono::milliseconds(1500));

    std::cout << "[SHUTDOWN]: Transmitting termination barrier to compute engine...\n";
    masterEngine.stop();

    // Join the GPU thread back to the main CPU thread contiguously
    if (computeWorker.joinable()) {
        computeWorker.join();
        std::cout << "[SHUTDOWN]: GPU Compute Worker Thread successfully synchronized and joined.\n";
    }

    std::cout << "\n============================================================\n";
    std::cout << " [SUCCESS]: Batch Cryptographic Processing Complete!        \n";
    std::cout << " Verify your encrypted video frames inside '" << output_dir << "/'.\n";
    std::cout << "============================================================\n";

    return 0;
}