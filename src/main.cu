#include <iostream>
#include <filesystem>
#include <vector>
#include <string>
#include <thread>
#include <algorithm>
#include <fstream>
#include <unordered_map>
#include <chrono>

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
        std::cerr << "    -> [chen VAULT]: Entered function." << std::endl;

        std::ifstream file("engine_keys.txt");
        std::string token;
        
        // MATHEMATICALLY STABLE DEFAULTS
        float k1 = 35.0f, k2 = 3.0f, k3 = 28.0f;
        float x = 0.1234567f, y = 0.5432198f, z = 0.9876543f; 
        int t = 1000;
        std::cerr << "    -> [chen VAULT]: Default vars set. Checking file..." << std::endl;

        if (file.is_open()) {
            // Using '>>' natively ignores all \r and \n formatting bugs!
            while (file >> token) {
                if (token == "[CHEN]") {
                    file >> k1 >> k2 >> k3 >> t >> x >> y >> z;
                    break;
                }
            }
        } else {
            std::cerr << "  [WARNING]: engine_keys.txt not found. Booting with default Chen seeds.\n";
        }
        std::cerr << "    -> [chen VAULT]: Returning struct (Danger Zone)..." << std::endl;

        return chenInitialArguments(k1, k2, k3, t, x, y, z);
    }

    lorenzInitialArguments getLorenzMasterKeys() {
        std::cerr << "    -> [LORENZ VAULT]: Entered function." << std::endl;
        std::ifstream file("engine_keys.txt");
        std::string token;
        
        float a = 10.0f, b = (8.0f / 3.0f), c = 46.0f, r = 2.0f;
        float x = 12.0f, y = 0.7194113f, z = 0.8156727f, w = 0.2946892f, step = 0.4389124f; 
        int t = 1500;

        std::cerr << "    -> [LORENZ VAULT]: Default vars set. Checking file..." << std::endl;
        if (file.is_open()) {
            while (file >> token) {
                if (token == "[LORENZ]") {
                    file >> a >> b >> c >> r >> x >> t >> y >> z >> w >> step;
                    break;
                }
            }
        } else {
            std::cerr << "    -> [WARNING]: engine_keys.txt not found. Using Lorenz defaults." << std::endl;
        }
        
        std::cerr << "    -> [LORENZ VAULT]: Returning struct (Danger Zone)..." << std::endl;
        return lorenzInitialArguments(a, b, c, r, x, t, y, z, w, step);
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
        std::sort(filepaths.begin(), filepaths.end());
        return filepaths;
    }

    void executeEncryption(const SystemConfig& config, bool runDifferentialAsset = false) {
        std::vector<std::string> targets = _fetchImageTargets(config.inputDir);
        if (targets.empty()) return;

        if (!fs::exists(config.outputDir)) fs::create_directories(config.outputDir);

        // std::endl forces Python to display the text immediately!
        std::cout << "[SYSTEM BOOT]: Igniting Pure Compute Encryption Engine..." << std::endl;
        
        int w = 0, h = 0, c = 0;
        unsigned char* probe = stbi_load(targets[0].c_str(), &w, &h, &c, 3);
        if(!probe) {
            std::cout << "[FATAL]: Failed to load initial probe image!" << std::endl;
            return;
        }
        stbi_image_free(probe); 
        
        imageData streamFormat { nullptr, w * h * 3, h, w, 3 };

        encryptionEngine* masterEncrypt = nullptr;
        try {
            chenInitialArguments cKeys = KeyVault::getChenMasterKeys();
            
            lorenzInitialArguments lKeys = KeyVault::getLorenzMasterKeys();
            
            masterEncrypt = new encryptionEngine(streamFormat, cKeys, lKeys);
                        
        } catch (const std::exception& e) {
            return; // Exit safely instead of crashing Windows!
        }

        std::cout << "  -> Engine Initialized successfully!" << std::endl;

        for (const auto& file : targets) {
            std::string original_filename = fs::path(file).filename().string();
            std::cout << "\n  [HOST]: Encrypting " << original_filename << "..." << std::endl;

            auto time_total_start = std::chrono::high_resolution_clock::now();

            auto time_io_start = std::chrono::high_resolution_clock::now();
            unsigned char* h_frameBytes = stbi_load(file.c_str(), &w, &h, &c, 3);
            if (!h_frameBytes) {
                std::cout << "  [ERROR]: Failed to load " << original_filename << std::endl;
                continue;
            }
            auto time_io_end = std::chrono::high_resolution_clock::now();
            float disk_io_ms = std::chrono::duration<float, std::milli>(time_io_end - time_io_start).count();

            std::cout << "  -> Firing GPU Kernels..." << std::endl;
            auto ciphers = masterEncrypt->encrypt(h_frameBytes, streamFormat.sizeOfImageFileInByte);
            std::cout << "  -> GPU Kernels Complete!" << std::endl;

            std::cout << "  -> Exporting Keystreams..." << std::endl;
            masterEncrypt->exportKeystreams(config.outputDir, streamFormat.sizeOfImageFileInByte);
            std::cout << "  -> Keystreams Exported!" << std::endl;

            time_io_start = std::chrono::high_resolution_clock::now();
            std::string main_out = config.outputDir + "/" + original_filename;
            stbi_write_png(main_out.c_str(), w, h, 3, ciphers.first, 0);

            std::string aux_out = config.outputDir + "/aux_" + original_filename;
            stbi_write_png(aux_out.c_str(), w, h, 3, ciphers.second, 0);
            time_io_end = std::chrono::high_resolution_clock::now();
            
            disk_io_ms += std::chrono::duration<float, std::milli>(time_io_end - time_io_start).count();

            auto time_total_end = std::chrono::high_resolution_clock::now();
            float total_end_to_end_ms = std::chrono::duration<float, std::milli>(time_total_end - time_total_start).count();

            std::cout << "      [HOST METRICS]: Disk I/O (stb_image)  : " << disk_io_ms << " ms" << std::endl;
            std::cout << "      [SYS METRICS] : Total End-to-End Time : " << total_end_to_end_ms << " ms" << std::endl;
            
            float size_in_mb = (w * h * 3) / (1024.0f * 1024.0f);
            float throughput_mb_s = size_in_mb / (total_end_to_end_ms / 1000.0f);
            std::cout << "      [PERFORMANCE] : Throughput            : " << throughput_mb_s << " MB/s" << std::endl;

            delete[] ciphers.first;
            delete[] ciphers.second;
            stbi_image_free(h_frameBytes);
        }
        
        std::cout << " [SUCCESS]: Encryption Complete! Saved -> '" << config.outputDir << "/'" << std::endl;
        delete masterEncrypt;
    }

    void executeDecryption(const SystemConfig& config) {
        std::vector<std::string> targets = _fetchImageTargets(config.inputDir);
        if (targets.empty()) return;

        if (!fs::exists(config.outputDir)) fs::create_directories(config.outputDir);

        std::cout << "[SYSTEM BOOT]: Igniting Pure Compute Decryption Engine...\n";
        
        int w = 0, h = 0, c = 0;
        unsigned char* probe = stbi_load(targets[0].c_str(), &w, &h, &c, 3);
        if(!probe) return;
        stbi_image_free(probe); 
        
        imageData streamFormat { nullptr, w * h * 3, h, w, 3 };

        decryptionEngine* masterDecrypt{nullptr};
        try {
            chenInitialArguments cKeys = KeyVault::getChenMasterKeys();
            
            lorenzInitialArguments lKeys = KeyVault::getLorenzMasterKeys();
            
            masterDecrypt = new decryptionEngine(streamFormat, cKeys, lKeys);
                        
        } catch (const std::exception& e) {
            return; // Exit safely instead of crashing Windows!
        }
        
        for (const auto& file : targets) {
            std::string original_filename = fs::path(file).filename().string();
            
            // Skip the auxiliary files during the main loop, we load them manually!
            if (original_filename.find("aux_") == 0) continue;

            std::string aux_filepath = config.inputDir + "/aux_" + original_filename;

            unsigned char* main_cipher = stbi_load(file.c_str(), &w, &h, &c, 3);
            unsigned char* aux_cipher = stbi_load(aux_filepath.c_str(), &w, &h, &c, 3);

            if (!main_cipher || !aux_cipher) {
                std::cerr << "  [ERROR]: Missing matching aux_ file for " << original_filename << "\n";
                if(main_cipher) stbi_image_free(main_cipher);
                continue;
            }

            std::cout << "  [HOST]: Decrypting " << original_filename << "...\n";

            // Trigger Engine
            unsigned char* plainText = masterDecrypt->decrypt(main_cipher, aux_cipher, streamFormat.sizeOfImageFileInByte);

            std::string out_path = config.outputDir + "/" + original_filename;
            stbi_write_png(out_path.c_str(), w, h, 3, plainText, 0);

            delete[] plainText;
            stbi_image_free(main_cipher);
            stbi_image_free(aux_cipher);
        }
        
        std::cout << " [SUCCESS]: Batch Decryption Complete! Output -> '" << config.outputDir << "/'\n";
    }
}


int main(int argc, char* argv[]) {
    
    SystemConfig config = ConfigManager::initialize(argc, argv);
    if (!config.isValid) return EXIT_FAILURE;

    if (config.mode == "encrypt") {
        ComputePipeline::executeEncryption(config, false);
    }  
    else if (config.mode == "decrypt") {
        ComputePipeline::executeDecryption(config);
    }

    return EXIT_SUCCESS;
}