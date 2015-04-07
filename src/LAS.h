#pragma once

#include <algorithm>
#include <vector>
#include <stdint.h>
#include <cublas_v2.h>
#include <random>

#include "Bicluster.h"
#include "LasException.h"

struct LasInstanceMemory;

class LAS
{
public:

    LAS(int cudaDeviceId = 0,
            cublasHandle_t cublasHandle = nullptr,
            uint16_t maxBiclusters = 10,
            uint16_t InvocationsPerBicluster = 1000,
            double scoreThreshold = 1.0);
    virtual ~LAS();

    uint32_t getInvocationsPerBicluster() const;
    void setInvocationsPerBicluster(uint32_t invocationsPerBicluster);

    uint16_t getMaxBiclusters() const;
    void setMaxBiclusters(uint16_t maxBiclusters);

    double getScoreThreshold() const;
    void setScoreThreshold(double scoreThreshold);

    // matrix - pointer to height x width column-major matrix
    std::vector<Bicluster> run(const double* matrix, uint16_t width,
                               uint16_t height)
                   throw (LasException, std::bad_alloc);

protected:

    void checkArguments(const double* matrix, uint32_t width, uint32_t height);

    virtual void initializeIteration(std::vector<uint16_t>& columnsPermutations,
                                     LasInstanceMemory& memory);

    std::pair<uint16_t, uint16_t> extractBicluster(
            LasInstanceMemory& memory,
            std::pair<uint32_t, double> biclusterIndexAndScore,
            std::vector<Bicluster>& biclusters);

    void subtractAverage(LasInstanceMemory& memory, uint32_t bestBicluster,
                         const std::pair<uint16_t, uint16_t>& size);

    void firstStageSearch(LasInstanceMemory& memory);
    void secondStageSearch(LasInstanceMemory& memory);

    int cudaDeviceId;
    cublasHandle_t cublasHandle;
    bool cublasCreated;

    uint16_t maxBiclusters;
    uint32_t invocationsPerBicluster;
    double scoreThreshold;

    std::random_device RANDOM_DEVICE;
    std::mt19937 MT19937_GENERATOR;
    std::uniform_real_distribution<> UNIFORM_REAL;
};

