#include <iostream>
#include <vector>
#include "chenChaoticSystem.h"

constexpr float h{0.00001f};

struct vec3 {
    float x; float y; float z;

    vec3 operator+(const vec3& r) const { return {x + r.x, y + r.y, z + r.z}; }
    vec3 operator*(float s) const       { return {x * s, y * s, z * s}; }
};

inline vec3 getChenDerivatives(const vec3& p, float a, float b, float c) {
    return {
        a * (p.y - p.x),
        (c - a) * p.x - p.x * p.z + c * p.y,
        p.x * p.y - b * p.z
    };
}

inline vec3 stepRK4Chen(const vec3& p, float a, float b, float c) {
    vec3 k1 = getChenDerivatives(p, a, b, c);
    vec3 k2 = getChenDerivatives(p + (k1 * (h * 0.5f)), a, b, c);
    vec3 k3 = getChenDerivatives(p + (k2 * (h * 0.5f)), a, b, c);
    vec3 k4 = getChenDerivatives(p + (k3 * h), a, b, c);

    return p + ((k1 + (k2 * 2.0f) + (k3 * 2.0f) + k4) * (h / 6.0f));
}

// ============================================================================

chenChaoticSystem::chenChaoticSystem(
    chenInitialArguments initialArgument,
    int requiredNoOfChaoticNumbers
) : m_initialArguments{initialArgument},
    m_sizeOfChaoticStream{requiredNoOfChaoticNumbers} 
{
    m_chaoticStreams.x = new float[m_sizeOfChaoticStream];
    m_chaoticStreams.y = new float[m_sizeOfChaoticStream];
    m_chaoticStreams.z = new float[m_sizeOfChaoticStream];
}

chenChaoticSystem::~chenChaoticSystem() {
    delete[] m_chaoticStreams.x;
    delete[] m_chaoticStreams.y;
    delete[] m_chaoticStreams.z;
}

void chenChaoticSystem::generate() {
    vec3 curr {
        m_initialArguments.x, m_initialArguments.y, m_initialArguments.z
    };

    const float a = m_initialArguments.a;
    const float b = m_initialArguments.b;
    const float c = m_initialArguments.c;

    // 1. Burn-in / Transient Discard Phase (Stabilizing the trajectory into chaos)
    for(int i = 0; i < m_initialArguments.initialIterationCount; ++i) {
        curr = stepRK4Chen(curr, a, b, c);
    }

    // 2. Pure ODE stream capture
    for(int i = 0; i < m_sizeOfChaoticStream; ++i) {
        curr = stepRK4Chen(curr, a, b, c);

        m_chaoticStreams.x[i] = curr.x;
        m_chaoticStreams.y[i] = curr.y;
        m_chaoticStreams.z[i] = curr.z;
    }
}

chaoticStreamChen chenChaoticSystem::getChaoticStreams() { 
    return m_chaoticStreams; 
}