#pragma once
#include <stdexcept>
#include <string>
#include "vec4.h"

template <typename T = double>
struct lorenzInitialArguments {
    T a; T b; T c; T d; T e;
    int initialIterationCount;
    T x; T y; T z; T w;

    lorenzInitialArguments(T a, T b, T c, T d, T e, int iterationCount, T x, T y, T z, T w)
        : a(a), b(b), c(c), d(d), e(e), initialIterationCount(iterationCount), x(x), y(y), z(z), w(w)
    {
        if (a != static_cast<T>(10.0)) {
            throw std::invalid_argument("Lorenz System Error: Parameter 'a' must be exactly 10.");
        }
        if (b != static_cast<T>(8.0) / static_cast<T>(3.0)) {
            throw std::invalid_argument("Lorenz System Error: Parameter 'b' must be exactly 8/3.");
        }
        if (c != static_cast<T>(46.0)) {
            throw std::invalid_argument("Lorenz System Error: Parameter 'c' must be exactly 46.");
        }
        if (d != static_cast<T>(2.0)) {
            throw std::invalid_argument("Lorenz System Error: Parameter 'd' must be exactly 2.");
        }
        if (e != static_cast<T>(12.0)) {
            throw std::invalid_argument("Lorenz System Error: Parameter 'e' must be exactly 12.");
        }
    }
};

template <typename T = double>
struct chaoticStreamLorenz {
    T* x{nullptr};
    T* y{nullptr};
    T* z{nullptr};
    T* w{nullptr};
};

template <typename T = double>
class lorenzChaoticSystem {
private:
    lorenzInitialArguments<T> m_initialArguments;
    int                       m_sizeOfChaoticStream;
    chaoticStreamLorenz<T>    m_chaoticStreams;
    T                         m_h;
    int                       m_decimationFactor;

    inline vec4_t<T> getHyperLorenzDerivatives(const vec4_t<T>& p, T a, T b, T c, T d, T e) const {
        return {
            a * (p.y - p.x),
            c * p.x - p.x * p.z - p.y + e * p.w,
            p.x * p.y - b * p.z,
            -d * p.y
        };
    }

    inline vec4_t<T> stepRK4HyperLorenz(const vec4_t<T>& p, T a, T b, T c, T d, T e, T h) const {
        vec4_t<T> k1 = getHyperLorenzDerivatives(p, a, b, c, d, e);
        vec4_t<T> k2 = getHyperLorenzDerivatives(p + (k1 * (h * static_cast<T>(0.5))), a, b, c, d, e);
        vec4_t<T> k3 = getHyperLorenzDerivatives(p + (k2 * (h * static_cast<T>(0.5))), a, b, c, d, e);
        vec4_t<T> k4 = getHyperLorenzDerivatives(p + (k3 * h), a, b, c, d, e);
        return p + ((k1 + (k2 * static_cast<T>(2.0)) + (k3 * static_cast<T>(2.0)) + k4) * (h / static_cast<T>(6.0)));
    }

public:
    lorenzChaoticSystem(lorenzInitialArguments<T> initialArgs, 
                        int requiredChaoticOutputCount, 
                        T hStep = static_cast<T>(0.02), 
                        int decimation = 100)
        : m_initialArguments(initialArgs),
          m_sizeOfChaoticStream(requiredChaoticOutputCount),
          m_h(hStep),
          m_decimationFactor(decimation)
    {
        m_chaoticStreams.x = new T[m_sizeOfChaoticStream];
        m_chaoticStreams.y = new T[m_sizeOfChaoticStream];
        m_chaoticStreams.z = new T[m_sizeOfChaoticStream];
        m_chaoticStreams.w = new T[m_sizeOfChaoticStream];
    }

    ~lorenzChaoticSystem() {
        delete[] m_chaoticStreams.x;
        delete[] m_chaoticStreams.y;
        delete[] m_chaoticStreams.z;
        delete[] m_chaoticStreams.w;
    }

    void generate() {
        vec4_t<T> curr {
            m_initialArguments.x, m_initialArguments.y,
            m_initialArguments.z, m_initialArguments.w
        };

        const T a = m_initialArguments.a;
        const T b = m_initialArguments.b;
        const T c = m_initialArguments.c;
        const T d = m_initialArguments.d;
        const T e = m_initialArguments.e;

        for (int i = 0; i < m_initialArguments.initialIterationCount; ++i) {
            curr = stepRK4HyperLorenz(curr, a, b, c, d, e, m_h);
        }

        for (int i = 0; i < m_sizeOfChaoticStream; ++i) {
            for (int skip = 0; skip < m_decimationFactor; ++skip) {
                curr = stepRK4HyperLorenz(curr, a, b, c, d, e, m_h);
            }
            m_chaoticStreams.x[i] = curr.x;
            m_chaoticStreams.y[i] = curr.y;
            m_chaoticStreams.z[i] = curr.z;
            m_chaoticStreams.w[i] = curr.w;
        }
    }

    chaoticStreamLorenz<T> getChaoticStream() const { return m_chaoticStreams; }
};