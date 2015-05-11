#pragma once

#include <fstream>
#include <gtest/gtest.h>
#include <cublas_v2.h>

#include "../../ErrorHandling.h"
#include "../../LAS.h"
#include "../../LasInstanceMemory.h"

class WholeAlgo : public ::testing::Test, public LAS
{
protected:

    WholeAlgo()
            : LAS(0, NULL),
              MatrixSize(loadMatrixSize()),
              goldNumberOfFoundBiclusters(loadGoldNumberOfFoundBiclusters()),
              matrix(NULL),
              iteration(1),
              goldColumnSet(goldNumberOfFoundBiclusters, MatrixSize.first,
                      MatrixLocation::Host),
              goldRowSet(MatrixSize.second, goldNumberOfFoundBiclusters,
                      MatrixLocation::Host),
              goldScores(goldNumberOfFoundBiclusters, 1, MatrixLocation::Host)
    {
        matrix = new double[MatrixSize.first * MatrixSize.second];
        goldColumnSet.allocate();
        goldRowSet.allocate();
        goldScores.allocate();

        std::string tmp;

        uint16_t numberOfBiclusters, invocationsPerBicluster;
        double scoreThreshold;

        std::ifstream infile("data/wholeAlgo.params.csv");
        assert(infile.is_open());

        for (int i = 0; i < 8; i++)
            std::getline(infile, tmp);
        infile >> numberOfBiclusters;

        for (int i = 0; i < 5; i++)
            std::getline(infile, tmp);
        infile >> invocationsPerBicluster;

        for (int i = 0; i < 5; i++)
            std::getline(infile, tmp);
        infile >> scoreThreshold;

        infile.close();
        setMaxBiclusters(numberOfBiclusters);
        setInvocationsPerBicluster(invocationsPerBicluster);
        setScoreThreshold(scoreThreshold);

        infile.open("data/wholeAlgo.matrix.csv");
        assert(infile.is_open());
        for (int i = 0; i < 5; i++)
            std::getline(infile, tmp);
        for (int i = 0; i < MatrixSize.second; i++)
        {
            for (int j = 0; j < MatrixSize.first; j++)
            {
                double value;
                infile >> value;
                matrix[j * MatrixSize.second + i] = value;
            }
        }
        infile.close();

        infile.open("data/wholeAlgo.goldRowSet.csv");
        assert(infile.is_open());
        for (int i = 0; i < 5; i++)
            std::getline(infile, tmp);
        for (int i = 0; i < goldRowSet.height; i++)
        {
            for (int j = 0; j < goldRowSet.width; j++)
            {
                uint32_t value;
                infile >> value;
                goldRowSet.data[j * goldRowSet.height + i] = value;
            }
        }
        infile.close();

        infile.open("data/wholeAlgo.goldColumnSet.csv");
        assert(infile.is_open());
        for (int i = 0; i < 5; i++)
            std::getline(infile, tmp);
        for (int i = 0; i < goldColumnSet.height; i++)
        {
            for (int j = 0; j < goldColumnSet.width; j++)
            {
                uint32_t value;
                infile >> value;
                goldColumnSet.data[j * goldColumnSet.height + i] = value;
            }
        }
        infile.close();

        infile.open("data/wholeAlgo.goldScores.csv");
        assert(infile.is_open());
        for (int i = 0; i < 5; i++)
            std::getline(infile, tmp);
        for (int i = 0; i < goldNumberOfFoundBiclusters; i++)
        {
            double value;
            infile >> value;
            goldScores.data[i] = value;
        }
        infile.close();
    }

    void initializeIteration(std::vector<uint16_t>& columnsPermutations,
                             LasInstanceMemory& memory)
    {
        std::ifstream infile(
                "data/wholeAlgo." + std::to_string(iteration) + ".init.csv");
        assert(infile.is_open());

        // generating initial submatrices sizes
        std::string tmp;
        for (int i = 0; i < 5; ++i)
            std::getline(infile, tmp);
        for (int i = 0; i < 2; ++i)
        {
            for (int j = 0; j < invocationsPerBicluster; ++j)
            {
                uint16_t size;
                infile >> size;
                memory.sizes.data[j * 2 + i] = size;
            }
        }

        memory.sizes.copyTo(memory.deviceSizes);

        // initializing permutations
        for (int i = 0; i < 7; ++i)
            std::getline(infile, tmp);
        for (int i = 0; i < memory.columnSet.height; ++i)
        {
            for (int j = 0; j < memory.columnSet.width; ++j)
            {
                double value;
                infile >> value;
                memory.columnSet.data[j * memory.columnSet.height + i] = value;
            }
        }
        memory.columnSet.copyTo(memory.deviceColumnSet);

        checkCudaErrors(
                cudaMemset(memory.deviceRowSet.data, 0,
                        memory.deviceRowSet.size()));

        iteration++;
    }

    virtual ~WholeAlgo()
    {
    }

    static std::pair<uint16_t, uint16_t> loadMatrixSize()
    {
        uint32_t M, N;

        std::ifstream infile("data/wholeAlgo.matrix.csv");
        assert(infile.is_open());
        std::string tmp;
        for (int i = 0; i < 4; i++)
            std::getline(infile, tmp);
        N = std::stoi(tmp.substr(tmp.rfind(' ')));
        std::getline(infile, tmp);
        M = std::stoi(tmp.substr(tmp.rfind(' ')));

        return std::make_pair(M, N);
    }

    static uint16_t loadGoldNumberOfFoundBiclusters()
    {
        uint16_t numberOfFoundBiclusters;

        std::ifstream infile("data/wholeAlgo.params.csv");
        assert(infile.is_open());
        std::string tmp;
        for (int i = 0; i < 3; i++)
            std::getline(infile, tmp);
        infile >> numberOfFoundBiclusters;

        return numberOfFoundBiclusters;
    }

    std::pair<uint16_t, uint16_t> MatrixSize;
    uint16_t goldNumberOfFoundBiclusters;

    double* matrix;
    uint32_t iteration;

    Matrix<uint8_t> goldColumnSet;
    Matrix<uint8_t> goldRowSet;
    Matrix<double> goldScores;
};

TEST_F(WholeAlgo, Works)
{
    auto result = run(matrix, MatrixSize.first, MatrixSize.second);
    checkCudaErrors(cudaGetLastError());

    EXPECT_EQ(result.size(), goldNumberOfFoundBiclusters);
    if (result.size() != goldNumberOfFoundBiclusters)
    {
        assert(!"Result sizes do not match.");
    }

    for (int i = 0; i < goldNumberOfFoundBiclusters; ++i)
    {

        EXPECT_NEAR(goldScores.data[i], result[i].score, 1e-10);

        for (int j = 0; j < MatrixSize.first; ++j)
        {
            EXPECT_EQ(goldColumnSet.data[i * goldColumnSet.height + j],
                    result[i].selectedColumns[j]);
        }

        for (int j = 0; j < MatrixSize.second; ++j)
        {
            EXPECT_EQ(goldRowSet.data[j * goldRowSet.height + i],
                    result[i].selectedRows[j]);
        }
    }
}

