#include <iostream>
#include <vector>
#include "lorenzHyperChaoticSystem.h"

constexpr float h{0.00001f};
constexpr int DECIMATION_FACTOR{50}; 

struct vec4 {
    float x; float y; float z; float w;
    vec4 operator+(const vec4& r) const { return {x + r.x, y + r.y, z + r.z, w + r.w}; }
    vec4 operator*(float s) const       { return {x * s, y * s, z * s, w * s}; }
};

inline vec4 getHyperLorenzDerivatives(const vec4& p, float a, float b, float c, float d, float e) {
    return {
        a * (p.y - p.x),
        c * p.x - p.x * p.z - p.y + e * p.w,
        p.x * p.y - b * p.z,
        -d * p.y
    };
}

inline vec4 stepRK4HyperLorenz(const vec4& p, float a, float b, float c, float d, float e) {
    vec4 k1 = getHyperLorenzDerivatives(p, a, b, c, d, e);
    vec4 k2 = getHyperLorenzDerivatives(p + (k1 * (h * 0.5f)), a, b, c, d, e);
    vec4 k3 = getHyperLorenzDerivatives(p + (k2 * (h * 0.5f)), a, b, c, d, e);
    vec4 k4 = getHyperLorenzDerivatives(p + (k3 * h), a, b, c, d, e);

    return p + ((k1 + (k2 * 2.0f) + (k3 * 2.0f) + k4) * (h / 6.0f));
}

// ============================================================================

lorenzChaoticSystem::lorenzChaoticSystem(
    lorenzInitialArguments initialArgument,
    int requriedNoOfChaoticNumbers
) : m_initialArguments{initialArgument},
    m_sizeOfChaoticStream{requriedNoOfChaoticNumbers} 
{
    m_chaoticStreams.x = new float[m_sizeOfChaoticStream];
    m_chaoticStreams.y = new float[m_sizeOfChaoticStream];
    m_chaoticStreams.z = new float[m_sizeOfChaoticStream];
    m_chaoticStreams.w = new float[m_sizeOfChaoticStream];
}

lorenzChaoticSystem::~lorenzChaoticSystem() {
    delete[] m_chaoticStreams.x;
    delete[] m_chaoticStreams.y;
    delete[] m_chaoticStreams.z;
    delete[] m_chaoticStreams.w;
}

void lorenzChaoticSystem::generate() {
    vec4 curr {
        m_initialArguments.x, m_initialArguments.y, 
        m_initialArguments.z, m_initialArguments.w
    };

    const float a = m_initialArguments.a;
    const float b = m_initialArguments.b;
    const float c = m_initialArguments.c;
    const float d = m_initialArguments.d;
    const float e = m_initialArguments.e;

    // 1. Burn-in
    for(int i = 0; i < m_initialArguments.initialIterationCount; ++i) {
        curr = stepRK4HyperLorenz(curr, a, b, c, d, e);
    }

    // 2. Pure ODE stream capture (NOW WITH DECIMATION)
    for(int i = 0; i < m_sizeOfChaoticStream; ++i) {
        
        // Skip 50 iterations between saves
        for(int skip = 0; skip < DECIMATION_FACTOR; ++skip) {
            curr = stepRK4HyperLorenz(curr, a, b, c, d, e);
        }

        m_chaoticStreams.x[i] = curr.x;
        m_chaoticStreams.y[i] = curr.y;
        m_chaoticStreams.z[i] = curr.z;
        m_chaoticStreams.w[i] = curr.w;
    }
}

chaoticStreamLorenz lorenzChaoticSystem::getChaoticStream() { 
    return m_chaoticStreams; 
}