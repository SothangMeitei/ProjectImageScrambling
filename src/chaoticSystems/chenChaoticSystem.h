#pragma once
#include <stdexcept>
#include <string>

struct chenInitialArguments {
    float a; float b; float c;
    int initialIterationCount;
    float x; float y; float z;

    chenInitialArguments(float a, float b, float c, int iterationCount, float x, float y, float z)
        : a(a), b(b), c(c), initialIterationCount(iterationCount), x(x), y(y), z(z) 
    {
        if (a != 35.0f) {
            throw std::invalid_argument("Chen System Error: Parameter 'a' must be exactly 35 to guarantee a chaotic stream.");
        }
        if (b != 3.0f) {
            throw std::invalid_argument("Chen System Error: Parameter 'b' must be exactly 3 to guarantee a chaotic stream.");
        }
        if (c <= 20.0f || c >= 28.4f) {
            throw std::invalid_argument("Chen System Error: Parameter 'c' must be strictly between 20 and 28.4 (exclusive).");
        }
    }
};

struct chaoticStreamChen {
    float* x;
    float* y;
    float* z;
};

class chenChaoticSystem {
    private:
        chenInitialArguments m_initialArguments;

        int                 m_sizeOfChaoticStream;
        chaoticStreamChen   m_chaoticStreams;

    public:
        chenChaoticSystem(chenInitialArguments initialArgs, int requiredChaoticOutputCount);
        ~chenChaoticSystem();

        void                generate();
        chaoticStreamChen   getChaoticStreams();
};