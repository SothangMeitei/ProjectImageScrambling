#include <iostream>
#include <filesystem>
#include <vector>
#include <string>
#include <thread>
#include <algorithm>
#include <fstream>
#include <unordered_map>

#define STB_IMAGE_IMPLEMENTATION
#include "../vendor/stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../vendor/stb/stb_image_write.h"

#include "encryptionEngine/encryptionEngine.h"
#include "encryptionEngine/decryptionEngine.h"
#include "chaoticSystems/chenChaoticSystem.h"
#include "chaoticSystems/lorenzHyperChaoticSystem.h"
#include "encryptionEngine/imageData.h"

namespace fs = std::filesystem;


struct SystemConfig {
    bool isValid;
    std::string mode;
    std::string inputDir;
    std::string outputDir;
};

namespace ConfigManager {
    std::string _trim(const std::string& str) {
        size_t first = str.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) return "";
        size_t last = str.find_last_not_of(" \t\r\n");
        return str.substr(first, (last - first + 1));
    }

    std::unordered_map<std::string, std::string> _loadOrGenerateTextConfig(const std::string& filename) {
        std::unordered_map<std::string, std::string> config;
        std::ifstream file(filename);
        
        if (!file.is_open()) {
            std::ofstream out(filename);
            out << "# DNA Cipher Engine - Directory Configuration\n\n";
            out << "[ENCRYPT]\nencrypt_src = assets\nencrypt_out = outputs\n\n";
            out << "[DECRYPT]\ndecrypt_src = outputs\ndecrypt_out = decrypted_outputs\n\n";
            out << "[BENCHMARK]\nbenchmark_src = assets\nbenchmark_out = benchmark_outputs\n";
            out.close();
            file.open(filename);
            std::cout << "[SYSTEM]: Generated default '" << filename << "' configuration file.\n";
        }

        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#' || line[0] == '[') continue;
            size_t delimPos = line.find('=');
            if (delimPos != std::string::npos) {
                config[_trim(line.substr(0, delimPos))] = _trim(line.substr(delimPos + 1));
            }
        }
        return config;
    }

    SystemConfig initialize(int argc, char* argv[]) {
        SystemConfig config { false, "", "", "" };
        if (argc != 2) {
            std::cerr << "Usage: ./engine <encrypt | decrypt | benchmark>\n";
            return config;
        }

        config.mode = argv[1];
        auto textConfig = _loadOrGenerateTextConfig("engine_config.txt");

        if (config.mode == "encrypt") {
            config.inputDir  = textConfig["encrypt_src"];
            config.outputDir = textConfig["encrypt_out"];
        } else if (config.mode == "decrypt") {
            config.inputDir  = textConfig["decrypt_src"];
            config.outputDir = textConfig["decrypt_out"];
        } else if (config.mode == "benchmark") {
            config.inputDir  = textConfig["benchmark_src"];
            config.outputDir = textConfig["benchmark_out"];
        } else {
            std::cerr << "[ERROR]: Unknown mode: " << config.mode << "\n";
            return config;
        }

        if (!fs::exists(config.inputDir)) {
            std::cerr << "[FATAL ERROR]: Input directory '" << config.inputDir << "' does not exist!\n";
            return config;
        }

        config.isValid = true;
        return config;
    }
}

namespace KeyVault {
    chenInitialArguments getChenMasterKeys() {
        return chenInitialArguments(35.0f, 3.0f, 28.0f, 1000, 0.1234567f, 0.5432198f, 0.9876543f);
    }

    lorenzInitialArguments getLorenzMasterKeys() {
        return lorenzInitialArguments(10.0f, (8.0f / 3.0f), 46.0f, 2.0f, 12.0f, 1500, 0.7194113f, 0.8156727f, 0.2946892f, 0.4389124f);
    }
}


namespace ComputePipeline {

    std::vector<std::string> _fetchImageTargets(const std::string& directory) {
        std::vector<std::string> filepaths;
        for (const auto& entry : fs::directory_iterator(directory)) {
            std::string path = entry.path().string();
            if (path.find(".png") != std::string::npos || path.find(".jpg") != std::string::npos) {
                filepaths.push_back(path);
            }
        }
        return filepaths;
    }

    template <typename EngineType>
    void _blockUntilQueueDrained(EngineType& engine) {
        while (engine.getRemainingQueueSize() > 0) {
            std::this_thread::yield(); 
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(250)); 
    }

    void executeEncryption(const SystemConfig& config, bool runDifferentialAsset = false) {
        std::vector<std::string> targets = _fetchImageTargets(config.inputDir);
        if (targets.empty()) {
            std::cerr << "[WARNING]: No .png or .jpg files found in '" << config.inputDir << "'\n";
            return;
        }

        std::cout << "[SYSTEM BOOT]: Igniting master encryptionEngine hardware...\n";
        imageData streamFormat { nullptr, 640, 360, 3, 640 * 360 * 3 };

        encryptionEngine masterEncrypt(streamFormat, KeyVault::getChenMasterKeys(), KeyVault::getLorenzMasterKeys(), config.outputDir);
        std::thread computeWorker(&encryptionEngine::run, &masterEncrypt);
        
        // Producer: Push frames to VRAM
        for (const auto& file : targets) {
            int w, h, c;
            unsigned char* h_frameBytes = stbi_load(file.c_str(), &w, &h, &c, 3);
            if (h_frameBytes) masterEncrypt.pushImageIntoQueueBuffer(h_frameBytes);
        }
        
        _blockUntilQueueDrained(masterEncrypt);
        masterEncrypt.stop();
        if (computeWorker.joinable()) computeWorker.join();
        
        // Benchmark Mode: Single-threaded injection of the differential asset
        if (runDifferentialAsset) {
            int w, h, c;
            unsigned char* h_diffFrame = stbi_load("diff_plain.png", &w, &h, &c, 3);
            if (h_diffFrame) {
                masterEncrypt.pushImageIntoQueueBuffer(h_diffFrame);
                masterEncrypt.run();
                stbi_image_free(h_diffFrame);
            }
        }

        std::cout << " [SUCCESS]: Encryption Complete! Saved -> '" << config.outputDir << "/'\n";
    }

    void executeDecryption(const SystemConfig& config) {
        std::vector<std::string> targets = _fetchImageTargets(config.inputDir);
        if (targets.empty()) return;

        std::cout << "[SYSTEM BOOT]: Igniting master decryptionEngine hardware...\n";
        imageData streamFormat { nullptr, 640, 360, 3, 640 * 360 * 3 };

        decryptionEngine masterDecrypt(streamFormat, KeyVault::getChenMasterKeys(), KeyVault::getLorenzMasterKeys(), config.outputDir);
        std::thread computeWorker(&decryptionEngine::run, &masterDecrypt);
        
        for (const auto& file : targets) {
            int w, h, c;
            unsigned char* h_frameBytes = stbi_load(file.c_str(), &w, &h, &c, 3);
            if (h_frameBytes) {
                imageData nextFrame = streamFormat;
                nextFrame.imagePixelValues = h_frameBytes;
                masterDecrypt.pushIntoTheQueue(nextFrame);
            }
        }
        
        std::cout << "[PRODUCER]: Ingestion complete. Draining VRAM compute arena...\n";
        _blockUntilQueueDrained(masterDecrypt);
        masterDecrypt.stop();
        if (computeWorker.joinable()) computeWorker.join();
        
        std::cout << " [SUCCESS]: Batch Decryption Complete! Output -> '" << config.outputDir << "/'\n";
    }
}


int main(int argc, char* argv[]) {
    
    SystemConfig config = ConfigManager::initialize(argc, argv);
    if (!config.isValid) return EXIT_FAILURE;

    if (config.mode == "encrypt") {
        ComputePipeline::executeEncryption(config, false);
    } 
    else if (config.mode == "benchmark") {
        ComputePipeline::executeEncryption(config, true);
    } 
    else if (config.mode == "decrypt") {
        ComputePipeline::executeDecryption(config);
    }

    return EXIT_SUCCESS;
}