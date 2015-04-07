#pragma once

#include <gtest/gtest.h>
#include <cublas_v2.h>

#include "../../ErrorHandling.h"
#include "../../LAS.h"

class Utilities : public ::testing::Test, public LAS
{
protected:

    Utilities()
            : LAS(0, NULL, 10, 10),
              MatrixSize(loadMatrixSize()),
              dummyMatrix(new double[MatrixSize.first * MatrixSize.second]),
              memory(dummyMatrix, MatrixSize.first, MatrixSize.second, 10),
              goldUpdatedMatrix(memory.matrix.width, memory.matrix.height,
                      MatrixLocation::Host),
              i(-1)
    {
        memory.init();
        goldUpdatedMatrix.allocate();

        std::string tmp;

        std::ifstream infile("data/subtractAverage.i.csv");
        assert(infile.is_open());
        for (int i = 0; i < 3; i++)
            std::getline(infile, tmp);
        infile >> this->i;
        infile.close();

        infile.open("data/subtractAverage.sizes.csv");
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

        infile.open("data/subtractAverage.matrix.csv");
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

        infile.open("data/subtractAverage.rowSet.csv");
        assert(infile.is_open());
        for (int i = 0; i < 5; i++)
            std::getline(infile, tmp);
        for (int i = 0; i < memory.rowSet.height; i++)
        {
            for (int j = 0; j < memory.rowSet.width; j++)
            {
                double value;
                infile >> value;
                memory.rowSet.data[j * memory.rowSet.height + i] = value;
            }
        }
        infile.close();
        memory.rowSet.copyTo(memory.deviceRowSet);

        infile.open("data/subtractAverage.columnSet.csv");
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

        infile.open("data/subtractAverage.goldUpdatedMatrix.csv");
        assert(infile.is_open());
        for (int i = 0; i < 5; i++)
            std::getline(infile, tmp);
        for (int i = 0; i < memory.Height; i++)
        {
            for (int j = 0; j < memory.Width; j++)
            {
                double value;
                infile >> value;
                goldUpdatedMatrix.data[j * memory.Height + i] = value;
            }
        }
        infile.close();
    }

    virtual ~Utilities()
    {
    }

    static std::pair<uint16_t, uint16_t> loadMatrixSize()
    {
        uint32_t M, N;

        std::ifstream infile("data/subtractAverage.matrix.csv");
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
    Matrix<double> goldUpdatedMatrix;
    uint16_t i;
};

TEST_F(Utilities, Works)
{

    auto biclusterAndScore = std::make_pair(i, 17.3);
    std::vector<Bicluster> biclusters;

    auto size = extractBicluster(memory, biclusterAndScore, biclusters);
    checkCudaErrors(cudaGetLastError());

    EXPECT_EQ(memory.sizes.data[i * 2], size.first);
    EXPECT_EQ(memory.sizes.data[i * 2 + 1], size.second);

    EXPECT_EQ(biclusters.size(), 1);
    EXPECT_EQ(biclusters[0].width, size.first);
    EXPECT_EQ(biclusters[0].height, size.second);
    EXPECT_EQ(biclusters[0].score, 17.3);
    for (uint32_t j = 0; j < memory.Height; ++j)
    {
        EXPECT_EQ(biclusters[0].selectedRows[j],
                (uint8_t )memory.rowSet.data[j * memory.InvocationsPerBicluster
                        + i]);
    }
    for (uint32_t j = 0; j < memory.Width; ++j)
    {
        EXPECT_EQ(biclusters[0].selectedColumns[j],
                (uint8_t )memory.columnSet.data[i * memory.Width + j]);
    }

    subtractAverage(memory, i, size);
    checkCudaErrors(cudaGetLastError());
    memory.deviceMatrix.copyTo(memory.matrix);
    for (int j = 0; j < memory.Width; j++)
    {
        for (int i = 0; i < memory.Height; i++)
        {
            EXPECT_NEAR(goldUpdatedMatrix.data[j * memory.Height + i],
                    memory.matrix.data[j * memory.Height + i], 1e-10);
        }
    }
}

