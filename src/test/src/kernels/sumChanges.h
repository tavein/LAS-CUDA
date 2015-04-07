#pragma once

#include "../../../kernels/kernels.h"
#include <gtest/gtest.h>


TEST(SumChanges, WorksOn10Number) {
    Matrix<uint32_t> changes(10, 1, MatrixLocation::Host);
    changes.allocate();
    uint32_t expectedSum = 0;
    for(int i = 0; i < 10; i++) {
        changes.data[i] = std::floor(std::pow(UNIFORM_REAL(MT19937_GENERATOR), 2)*1000) + 1;
        expectedSum += changes.data[i];
    }

    Matrix<uint32_t> deviceChanges(10, 1, MatrixLocation::Device);
    deviceChanges.allocate();
    changes.copyTo(deviceChanges);

    uint32_t result = reduceChanges(deviceChanges);
    checkCudaErrors(cudaGetLastError());

    EXPECT_EQ(expectedSum, result);
}

TEST(SumChanges, WorksOn100Number) {
    Matrix<uint32_t> changes(100, 1, MatrixLocation::Host);
    changes.allocate();
    uint32_t expectedSum = 0;
    for(int i = 0; i < 100; i++) {
        changes.data[i] = std::floor(std::pow(UNIFORM_REAL(MT19937_GENERATOR), 2)*1000) + 1;
        expectedSum += changes.data[i];
    }

    Matrix<uint32_t> deviceChanges(100, 1, MatrixLocation::Device);
    deviceChanges.allocate();
    changes.copyTo(deviceChanges);

    uint32_t result = reduceChanges(deviceChanges);
    checkCudaErrors(cudaGetLastError());

    EXPECT_EQ(expectedSum, result);
}


TEST(SumChanges, WorksOn739Number) {
    Matrix<uint32_t> changes(739, 1, MatrixLocation::Host);
    changes.allocate();
    uint32_t expectedSum = 0;
    for(int i = 0; i < 739; i++) {
        changes.data[i] = std::floor(std::pow(UNIFORM_REAL(MT19937_GENERATOR), 2)*1000) + 1;
        expectedSum += changes.data[i];
    }

    Matrix<uint32_t> deviceChanges(739, 1, MatrixLocation::Device);
    deviceChanges.allocate();
    changes.copyTo(deviceChanges);

    uint32_t result = reduceChanges(deviceChanges);
    checkCudaErrors(cudaGetLastError());

    EXPECT_EQ(expectedSum, result);
}

