#pragma once

#include <gtest/gtest.h>
#include <cublas_v2.h>

#include "../../ErrorHandling.h"
#include "../../LAS.h"

class SecondStage : public ::testing::Test, public LAS
{
protected:

    SecondStage()
            : LAS(0, NULL, 10, 10),
              MatrixSize(loadMatrixSize()),
              dummyMatrix(new double[MatrixSize.first * MatrixSize.second]),
              memory(dummyMatrix, MatrixSize.first, MatrixSize.second, 10),
              goldColumnSet(memory.columnSet.width, memory.columnSet.height,
                      MatrixLocation::Host),
              goldRowSet(memory.rowSet.width, memory.rowSet.height,
                      MatrixLocation::Host),
              goldFinalScore(memory.InvocationsPerBicluster, 1,
                      MatrixLocation::Host)
    {
        memory.init();
        goldColumnSet.allocate();
        goldRowSet.allocate();
        goldFinalScore.allocate();

        std::string tmp;

        std::ifstream infile("data/secondStage.sizes.csv");
        assert(infile.is_open());
        for (int i = 0; i < 5; i++)
            std::getline(infile, tmp);
        for (int i = 0; i < memory.InvocationsPerBicluster; i++)
        {
            uint16_t width, height;
            infile >> width >> height;
            memory.sizes.data[i * 2] = width;
            memory.sizes.data[i * 2 + 1] = height;
        }
        infile.close();
        memory.sizes.copyTo(memory.deviceSizes);

        infile.open("data/secondStage.matrix.csv");
        assert(infile.is_open());
        for (int i = 0; i < 5; i++)
            std::getline(infile, tmp);
        for (int i = 0; i < memory.Height; i++)
        {
            for (int j = 0; j < memory.Width; j++)
            {
                double value;
                infile >> value;
                memory.matrix.data[j * memory.Height + i] = value;
            }
        }
        infile.close();
        memory.matrix.copyTo(memory.deviceMatrix);

        infile.open("data/secondStage.columnSet.csv");
        assert(infile.is_open());
        for (int i = 0; i < 5; i++)
            std::getline(infile, tmp);
        for (int i = 0; i < memory.columnSet.height; i++)
        {
            for (int j = 0; j < memory.columnSet.width; j++)
            {
                double value;
                infile >> value;
                memory.columnSet.data[j * memory.columnSet.height + i] = value;
            }
        }
        infile.close();
        memory.columnSet.copyTo(memory.deviceColumnSet);

        infile.open("data/secondStage.goldRowSet.csv");
        assert(infile.is_open());
        for (int i = 0; i < 5; i++)
            std::getline(infile, tmp);
        for (int i = 0; i < goldRowSet.height; i++)
        {
            for (int j = 0; j < goldRowSet.width; j++)
            {
                double value;
                infile >> value;
                goldRowSet.data[j * goldRowSet.height + i] = value;
            }
        }
        infile.close();

        infile.open("data/secondStage.goldColumnSet.csv");
        assert(infile.is_open());
        for (int i = 0; i < 5; i++)
            std::getline(infile, tmp);
        for (int i = 0; i < goldColumnSet.height; i++)
        {
            for (int j = 0; j < goldColumnSet.width; j++)
            {
                double value;
                infile >> value;
                goldColumnSet.data[j * goldColumnSet.height + i] = value;
            }
        }
        infile.close();

        infile.open("data/secondStage.goldFinalScore.csv");
        assert(infile.is_open());
        for (int i = 0; i < 5; i++)
            std::getline(infile, tmp);
        for (int i = 0; i < memory.InvocationsPerBicluster; i++)
        {
            double value;
            infile >> value;
            goldFinalScore.data[i] = value;
        }
        infile.close();
    }

    virtual ~SecondStage()
    {
    }

    static std::pair<uint16_t, uint16_t> loadMatrixSize()
    {
        uint32_t M, N;

        std::ifstream infile("data/secondStage.matrix.csv");
        assert(infile.is_open());
        std::string tmp;
        for (int i = 0; i < 4; i++)
            std::getline(infile, tmp);
        N = std::stoi(tmp.substr(tmp.rfind(' ')));
        std::getline(infile, tmp);
        M = std::stoi(tmp.substr(tmp.rfind(' ')));

        return std::make_pair(M, N);
    }

    std::pair<uint16_t, uint16_t> MatrixSize;
    double* dummyMatrix;

    LasInstanceMemory memory;

    Matrix<double> goldColumnSet;
    Matrix<double> goldRowSet;
    Matrix<double> goldFinalScore;
};

TEST_F(SecondStage, Works)
{
    secondStageSearch(memory);
    checkCudaErrors(cudaGetLastError());

    memory.deviceRowSet.copyTo(memory.rowSet);
    memory.deviceColumnSet.copyTo(memory.columnSet);

    for (int j = 0; j < memory.rowSet.width; j++)
    {
        for (int i = 0; i < memory.rowSet.height; i++)
        {
            EXPECT_EQ(goldRowSet.data[j * memory.rowSet.height + i],
                    memory.rowSet.data[j * memory.rowSet.height + i]);
        }
    }

    for (int j = 0; j < memory.columnSet.width; j++)
    {
        for (int i = 0; i < memory.columnSet.height; i++)
        {
            EXPECT_EQ(goldColumnSet.data[j * memory.columnSet.height + i],
                    memory.columnSet.data[j * memory.columnSet.height + i]);
        }
    }

    Matrix<double> biclusterScores(memory.InvocationsPerBicluster, 1,
            MatrixLocation::Host);
    biclusterScores.allocate();
    memory.deviceBiclusterScores.copyTo(biclusterScores);
    for (int i = 0; i < memory.InvocationsPerBicluster; i++)
    {
        EXPECT_NEAR(goldFinalScore.data[i], biclusterScores.data[i], 1e-10);
    }

}

