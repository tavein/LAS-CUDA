#pragma once

#include "../../../kernels/kernels.h"
#include <gtest/gtest.h>

#include <random>

TEST(MaxScore, WorksOn10Number)
{
    Matrix<double> scores(10, 1, MatrixLocation::Host);
    scores.allocate();

    uint32_t max = -1;
    double maxScoreValue = -INFINITY;
    for (int i = 0; i < 10; i++)
    {
        scores.data[i] = 100 * UNIFORM_REAL(MT19937_GENERATOR);

        if (scores.data[i] > maxScoreValue)
        {
            maxScoreValue = scores.data[i];
            max = i;
        }
    }

    Matrix<double> deviceScores(10, 1, MatrixLocation::Device);
    deviceScores.allocate();
    scores.copyTo(deviceScores);

    auto result = maxScore(deviceScores);
    checkCudaErrors(cudaGetLastError());

    EXPECT_EQ(max, result.first);
    EXPECT_EQ(maxScoreValue, result.second);
}

TEST(MaxScore, WorksOn100Number)
{
    Matrix<double> scores(100, 1, MatrixLocation::Host);
    scores.allocate();

    uint32_t max = -1;
    double maxScoreValue = -INFINITY;
    for (int i = 0; i < 100; i++)
    {
        scores.data[i] = UNIFORM_REAL(MT19937_GENERATOR);

        if (scores.data[i] > maxScoreValue)
        {
            maxScoreValue = scores.data[i];
            max = i;
        }
    }

    Matrix<double> deviceScores(100, 1, MatrixLocation::Device);
    deviceScores.allocate();
    scores.copyTo(deviceScores);

    auto result = maxScore(deviceScores);
    checkCudaErrors(cudaGetLastError());

    EXPECT_EQ(max, result.first);
    EXPECT_EQ(maxScoreValue, result.second);
}

TEST(MaxScore, WorksOn837Number)
{
    Matrix<double> scores(837, 1, MatrixLocation::Host);
    scores.allocate();

    uint32_t max = -1;
    double maxScoreValue = -INFINITY;
    for (int i = 0; i < 837; i++)
    {
        scores.data[i] = UNIFORM_REAL(MT19937_GENERATOR);

        if (scores.data[i] > maxScoreValue)
        {
            maxScoreValue = scores.data[i];
            max = i;
        }
    }

    Matrix<double> deviceScores(837, 1, MatrixLocation::Device);
    deviceScores.allocate();
    scores.copyTo(deviceScores);

    auto result = maxScore(deviceScores);
    checkCudaErrors(cudaGetLastError());

    EXPECT_EQ(max, result.first);
    EXPECT_EQ(maxScoreValue, result.second);
}
