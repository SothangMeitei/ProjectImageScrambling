#pragma once
#include <stdexcept>
#include <string>
#include "vec3.h"

template <typename T = double>
struct chenInitialArguments {
    T a; T b; T c;
    int initialIterationCount;
    T x; T y; T z;

    chenInitialArguments(T a, T b, T c, int iterationCount, T x, T y, T z)
        : a(a), b(b), c(c), initialIterationCount(iterationCount), x(x), y(y), z(z)
    {
        if (a != static_cast<T>(35.0)) {
            throw std::invalid_argument("Chen System Error: Parameter 'a' must be exactly 35 to guarantee a chaotic stream.");
        }
        if (b != static_cast<T>(3.0)) {
            throw std::invalid_argument("Chen System Error: Parameter 'b' must be exactly 3 to guarantee a chaotic stream.");
        }
        if (c <= static_cast<T>(20.0) || c >= static_cast<T>(28.4)) {
            throw std::invalid_argument("Chen System Error: Parameter 'c' must be strictly between 20 and 28.4 (exclusive).");
        }
    }
};

template <typename T = double>
struct chaoticStreamChen {
    T* x{nullptr};
    T* y{nullptr};
    T* z{nullptr};
};

template <typename T = double>
class chenChaoticSystem {
private:
    chenInitialArguments<T> m_initialArguments;
    int                     m_sizeOfChaoticStream;
    chaoticStreamChen<T>    m_chaoticStreams;
    T                       m_h;
    int                     m_decimationFactor;

    inline vec3_t<T> getChenDerivatives(const vec3_t<T>& p, T a, T b, T c) const {
        return {
            a * (p.y - p.x),
            (c - a) * p.x - p.x * p.z + c * p.y,
            p.x * p.y - b * p.z
        };
    }

    inline vec3_t<T> stepRK4Chen(const vec3_t<T>& p, T a, T b, T c, T h) const {
        vec3_t<T> k1 = getChenDerivatives(p, a, b, c);
        vec3_t<T> k2 = getChenDerivatives(p + (k1 * (h * static_cast<T>(0.5))), a, b, c);
        vec3_t<T> k3 = getChenDerivatives(p + (k2 * (h * static_cast<T>(0.5))), a, b, c);
        vec3_t<T> k4 = getChenDerivatives(p + (k3 * h), a, b, c);
        return p + ((k1 + (k2 * static_cast<T>(2.0)) + (k3 * static_cast<T>(2.0)) + k4) * (h / static_cast<T>(6.0)));
    }

public:
    chenChaoticSystem(chenInitialArguments<T> initialArgs, 
                       int requiredChaoticOutputCount, 
                       T hStep = static_cast<T>(0.00001), 
                       int decimation = 50)
        : m_initialArguments(initialArgs),
          m_sizeOfChaoticStream(requiredChaoticOutputCount),
          m_h(hStep),
          m_decimationFactor(decimation)
    {
        m_chaoticStreams.x = new T[m_sizeOfChaoticStream];
        m_chaoticStreams.y = new T[m_sizeOfChaoticStream];
        m_chaoticStreams.z = new T[m_sizeOfChaoticStream];
    }

    ~chenChaoticSystem() {
        delete[] m_chaoticStreams.x;
        delete[] m_chaoticStreams.y;
        delete[] m_chaoticStreams.z;
    }

    void generate() {
        vec3_t<T> curr {
            m_initialArguments.x, m_initialArguments.y, m_initialArguments.z
        };
        const T a = m_initialArguments.a;
        const T b = m_initialArguments.b;
        const T c = m_initialArguments.c;

        // 1. Burn-in Phase
        for (int i = 0; i < m_initialArguments.initialIterationCount; ++i) {
            curr = stepRK4Chen(curr, a, b, c, m_h);
        }

        // 2. Continuous ODE stream capture with decimation
        for (int i = 0; i < m_sizeOfChaoticStream; ++i) {
            for (int skip = 0; skip < m_decimationFactor; ++skip) {
                curr = stepRK4Chen(curr, a, b, c, m_h);
            }
            m_chaoticStreams.x[i] = curr.x;
            m_chaoticStreams.y[i] = curr.y;
            m_chaoticStreams.z[i] = curr.z;
        }
    }

    chaoticStreamChen<T> getChaoticStreams() const { return m_chaoticStreams; }
};