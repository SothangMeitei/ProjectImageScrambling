
#pragma once
#include <cuda_runtime.h>
#include"encryptionEngine.h"

#include "../vendor/stb/stb_image.h"
#include "../vendor/stb/stb_image_write.h"

encryptionEngine::encryptionEngine(){
    isRunning = true;
    isPaused = false;
}