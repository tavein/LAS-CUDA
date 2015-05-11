#pragma once

#include "../../../kernels/kernels.h"
#include <gtest/gtest.h>

#include <random>

static void MaxScoreWordOnN(uint32_t N) {
    Matrix<double> scores(N, 1, MatrixLocation::Host);
    scores.allocate();

    uint32_t max = -1;
    double maxScoreValue = -INFINITY;
    for (int i = 0; i < N; i++)
    {
        scores.data[i] = 100 * UNIFORM_REAL(MT19937_GENERATOR);

        if (scores.data[i] > maxScoreValue)
        {
            maxScoreValue = scores.data[i];
            max = i;
        }
    }

    Matrix<double> deviceScores(N, 1, MatrixLocation::Device);
    deviceScores.allocate();
    scores.copyTo(deviceScores);

    auto result = maxScore(deviceScores);
    checkCudaErrors(cudaGetLastError());

    EXPECT_EQ(max, result.first);
    EXPECT_EQ(maxScoreValue, result.second);
}

TEST(MaxScore, WorksOn10Number)
{
    MaxScoreWordOnN(10);
}

TEST(MaxScore, WorksOn100Number)
{
    MaxScoreWordOnN(100);
}

TEST(MaxScore, WorksOn837Number)
{
    MaxScoreWordOnN(837);
}
