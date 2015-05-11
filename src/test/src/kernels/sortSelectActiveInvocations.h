#pragma once

#include "../../../kernels/kernels.h"
#include <gtest/gtest.h>

#include <random>

void SortSelectActiveInvocationsWorksOnN(uint32_t N) {

    Matrix<uint32_t> changes(N, 1, MatrixLocation::Host);
    changes.allocate();

    uint32_t greaterThanZero = 0;
    for(int i = 0; i < N; ++i) {
        changes.data[i] = UNIFORM_INT(MT19937_GENERATOR);
        greaterThanZero += int(changes.data[i] > 0);
    }

    Matrix<uint32_t> deviceChanges(N, 1, MatrixLocation::Device);
    deviceChanges.allocate();
    changes.copyTo(deviceChanges);

    Matrix<uint16_t> devicePermutation(N, 1, MatrixLocation::Device);
    devicePermutation.allocate();

    auto result = sortSelectActiveInvocations(deviceChanges, devicePermutation, N);
    checkCudaErrors(cudaGetLastError());

    EXPECT_EQ(greaterThanZero, result);

    Matrix<uint16_t> permutation(N, 1, MatrixLocation::Host);
    permutation.allocate();
    devicePermutation.copyTo(permutation);

    Matrix<uint32_t> otherChanges(N, 1, MatrixLocation::Host);
    otherChanges.allocate();

    for(int i = 0; i < N; ++i) {
        otherChanges.data[i] = changes.data[permutation.data[i]];
    }

    deviceChanges.copyTo(changes);

    for(int i = 0; i < N; ++i) {
        if (i > 0) {
            EXPECT_GE(changes.data[i - 1], changes.data[i]);
        }
        EXPECT_EQ(otherChanges.data[i], changes.data[i]);
    }
}

TEST(SortSelectActiveInvocations, WorksOn100Number)
{
    SortSelectActiveInvocationsWorksOnN(100);
}


TEST(SortSelectActiveInvocations, WorksOn1000Number)
{
    SortSelectActiveInvocationsWorksOnN(1000);
}

TEST(SortSelectActiveInvocations, WorksOn8348Number)
{
    SortSelectActiveInvocationsWorksOnN(8348);
}
