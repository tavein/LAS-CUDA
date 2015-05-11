#pragma once

#include <stdint.h>

#include "Matrix.h"

// Nice struct to pass to device code
struct DeviceMemory
{
    double* ColumnSet = nullptr;

    double* RowSet = nullptr;

    double* Sums = nullptr;

    double* Scores = nullptr;

    double* gammaln = nullptr;

    uint16_t* Sizes = nullptr;

    uint32_t* ColumnChanges = nullptr;

    uint32_t* RowChanges = nullptr;
};

// Structure that holds all memory for single LAS invocation
struct LasInstanceMemory
{
    static uint32_t nextPowerOfTwo(uint32_t i);

    LasInstanceMemory(const double* matrix, const uint32_t& Width,
                      const uint32_t& Height,
                      const uint32_t& InvocationsPerBicluster);
    ~LasInstanceMemory();

    void init();

    LasInstanceMemory& operator =(const LasInstanceMemory&) = delete;
    LasInstanceMemory(const LasInstanceMemory&) = delete;

    const uint32_t Width;
    const uint32_t NextPowerOfTwoAfterWidth;
    const uint32_t Height;
    const uint32_t NextPowerOfTwoAfterHeight;
    const uint32_t MaxDimension;
    const uint32_t InvocationsPerBicluster;

    Matrix<double> matrix;
    Matrix<double> deviceMatrix;

    // Width rows x InvocationsPerBicluster columns binary matrix
    // representing set of selected columns, and it's device counterpart
    Matrix<double> columnSet;
    Matrix<double> deviceColumnSet;

    // InvocationsPerBicluster rows x Height columns binary matrix
    // representing set of selected rows, and it's device counterpart
    Matrix<double> rowSet;
    Matrix<double> deviceRowSet;

    // utility matrix to keep result of matrix multiplication
    // of set matrices with original matrix
    Matrix<double> sums;
    Matrix<double> deviceSums;

    // LAS scores of biclusters
    Matrix<double> deviceBiclusterScores;

    // sizes of biclusters
    Matrix<uint16_t> sizes;
    Matrix<uint16_t> deviceSizes;

    // array gammaln function values computed at integer points
    // (deviceLgamma[i] = gammaln(i))
    Matrix<double> deviceLgamma;

    // holds number of changes to sets during algorithm
    // stages
    Matrix<uint32_t> deviceColumnChanges;
    Matrix<uint32_t> deviceRowChanges;

    Matrix<uint16_t> deviceInvocationsPermutation;

    DeviceMemory deviceMemory;
};

