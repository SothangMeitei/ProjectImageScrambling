
#pragma once
#include <stdexcept>
#include <string>

struct lorenzInitialArguments {
    float a; float b; float c; float d; float e;
    int initialIterationCount;
    float x; float y; float z; float w;

    lorenzInitialArguments(float a, float b, float c, float d, float e, int iterationCount, float x, float y, float z , float w)
        : a(a), b(b), c(c), d(d) , e(e) , initialIterationCount(iterationCount), x(x), y(y), z(z) , w(w) 
    {
        if (a != 10.0f) {
            throw std::invalid_argument("Chen System Error: Parameter 'a' must be exactly 10 to guarantee a chaotic stream.");
        }
        if (b != 8.0f / 3.0f) {
            throw std::invalid_argument("Chen System Error: Parameter 'b' must be exactly 8/3 to guarantee a chaotic stream.");
        }
        if (c != 46.0f) {
            throw std::invalid_argument("Chen System Error: Parameter 'c' must be exactly 46 to guarantee a chaotic stream.");
        }
        if (d != 2.0f) {
            throw std::invalid_argument("Chen System Error: Parameter 'd' must be exactly 2 to guarantee a chaotic stream.");
        }
        if (e != 12.0f) {
            throw std::invalid_argument("Chen System Error: Parameter 'e' must be exactly 12 to guarantee a chaotic stream.");
        }
    }
};

struct chaoticStreamLorenz {
    float* x;
    float* y;
    float* z;
    float* w;
};

class lorenzChaoticSystem{
    private:
        lorenzInitialArguments m_initialArguments;

        int                     m_sizeOfChaoticStream;
        chaoticStreamLorenz     m_chaoticStreams;

    public:
        lorenzChaoticSystem(lorenzInitialArguments , int );

        //there are no getters or setters for the initial arguments as once the chen system has been built 
        //there is no point in altering the initial arguments for the stream generation

        ~lorenzChaoticSystem();

        void            generate();
        chaoticStreamLorenz  getChaoticStream();
};