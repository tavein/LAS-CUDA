#pragma once

#include <fstream>

#include "../../../LasInstanceMemory.h"
#include "../../../kernels/kernels.h"
#include <gtest/gtest.h>    

class SortSelectWidthAndColumns : public ::testing::Test
{
protected:

    SortSelectWidthAndColumns()
            : MatrixSize(loadMatrixSize()),
              dummyMatrixData(new double[MatrixSize.first * MatrixSize.second]),
              memory(dummyMatrixData, MatrixSize.first, MatrixSize.second, 10),
              goldColumnSet(memory.columnSet.width, memory.columnSet.height,
                      MatrixLocation::Host),
              goldChanges(memory.deviceColumnChanges.width,
                      memory.deviceColumnChanges.height, MatrixLocation::Host),
              goldNewSizes(memory.sizes.width, memory.sizes.height,
                      MatrixLocation::Host),
              goldMaxScores(memory.InvocationsPerBicluster, 1,
                      MatrixLocation::Host)
    {
        memory.init();
        goldColumnSet.allocate();
        goldChanges.allocate();
        goldNewSizes.allocate();
        goldMaxScores.allocate();
        std::string tmp;

        checkCudaErrors(
                cudaMemset(memory.deviceRowChanges.begin(), 1,
                        memory.deviceRowChanges.size()));
        checkCudaErrors(
                cudaMemset(memory.deviceColumnChanges.begin(), 0,
                        memory.deviceColumnChanges.size()));
        checkCudaErrors(
                cudaMemset(memory.deviceColumnSet.data, 0,
                        memory.deviceColumnSet.size()));

        std::ifstream infile("data/sortSelectWidthAndColumns.sums.csv");
        assert(infile.is_open());
        for (int i = 0; i < 5; i++)
            std::getline(infile, tmp);
        for (int i = 0; i < memory.InvocationsPerBicluster; i++)
        {
            for (int j = 0; j < memory.Width; j++)
            {
                double value;
                infile >> value;
                memory.sums.data[j * memory.InvocationsPerBicluster + i] =
                        value;
            }
        }
        infile.close();
        memory.sums.copyTo(memory.deviceSums);

        infile.open("data/sortSelectWidthAndColumns.sizes.csv");
        assert(infile.is_open());
        assert(memory.sizes.width == memory.InvocationsPerBicluster);
        assert(memory.sizes.height == 2);
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

        infile.open("data/sortSelectWidthAndColumns.goldColumnSet.csv");
        assert(infile.is_open());
        for (int i = 0; i < 5; i++)
            std::getline(infile, tmp);
        for (int i = 0; i < memory.columnSet.height; i++)
        {
            for (int j = 0; j < memory.columnSet.width; j++)
            {
                double value;
                infile >> value;
                goldColumnSet.data[j * memory.columnSet.height + i] = value;
            }
        }
        infile.close();

        infile.open("data/sortSelectWidthAndColumns.goldChanges.csv");
        assert(infile.is_open());
        for (int i = 0; i < 5; i++)
            std::getline(infile, tmp);
        for (int i = 0; i < memory.InvocationsPerBicluster; i++)
        {
            uint32_t value;
            infile >> value;
            goldChanges.data[i] = value;
        }
        infile.close();

        infile.open("data/sortSelectWidthAndColumns.goldNewSizes.csv");
        assert(infile.is_open());
        for (int i = 0; i < 5; i++)
            std::getline(infile, tmp);
        for (int i = 0; i < memory.InvocationsPerBicluster; i++)
        {
            uint16_t width, height;
            infile >> width >> height;
            goldNewSizes.data[i * 2] = width;
            goldNewSizes.data[i * 2 + 1] = height;
        }
        infile.close();

        infile.open("data/sortSelectWidthAndColumns.goldMaxScores.csv");
        assert(infile.is_open());
        for (int i = 0; i < 5; i++)
            std::getline(infile, tmp);
        for (int i = 0; i < memory.InvocationsPerBicluster; i++)
        {
            double score;
            infile >> score;
            goldMaxScores.data[i] = score;
        }
        infile.close();
    }

    virtual ~SortSelectWidthAndColumns()
    {
    }

    static std::pair<uint16_t, uint16_t> loadMatrixSize()
    {
        uint32_t M, N;

        std::ifstream infile("data/sortSelectWidthAndColumns.matrixSize.csv");
        assert(infile.is_open());
        std::string tmp;
        for (int i = 0; i < 3; i++)
            std::getline(infile, tmp);
        infile >> M;
        for (int i = 0; i < 5; i++)
            std::getline(infile, tmp);
        infile >> N;

        return std::make_pair(M, N);
    }

    std::pair<uint16_t, uint16_t> MatrixSize;
    double* dummyMatrixData;

    LasInstanceMemory memory;

    Matrix<double> goldColumnSet;
    Matrix<uint32_t> goldChanges;
    Matrix<uint16_t> goldNewSizes;
    Matrix<double> goldMaxScores;
};

TEST_F(SortSelectWidthAndColumns, Works)
{
    sortSelectWidthAndColumns(memory, memory.InvocationsPerBicluster);
    checkCudaErrors(cudaGetLastError());

    memory.deviceColumnSet.copyTo(memory.columnSet);
    for (int j = 0; j < memory.deviceColumnSet.width; j++)
    {
        for (int i = 0; i < memory.deviceColumnSet.height; i++)
        {
            EXPECT_EQ(goldColumnSet.data[j * memory.deviceColumnSet.height + i],
                    memory.columnSet.data[j * memory.deviceColumnSet.height + i]);
        }
    }

    Matrix<uint32_t> changes(memory.deviceColumnChanges.width,
            memory.deviceColumnChanges.height, MatrixLocation::Host);
    changes.allocate();
    memory.deviceColumnChanges.copyTo(changes);
    for (int i = 0; i < memory.InvocationsPerBicluster; i++)
    {
        EXPECT_EQ(goldChanges.data[i], changes.data[i]);
    }

    memset(memory.sizes.data, 0, memory.sizes.size());
    memory.deviceSizes.copyTo(memory.sizes);
    for (int i = 0; i < memory.InvocationsPerBicluster; i++)
    {
        EXPECT_EQ(goldNewSizes.data[i * 2 + 0], memory.sizes.data[i * 2 + 0]);
        EXPECT_EQ(goldNewSizes.data[i * 2 + 1], memory.sizes.data[i * 2 + 1]);
    }

    Matrix<double> biclusterScores(memory.InvocationsPerBicluster, 1,
            MatrixLocation::Host);
    biclusterScores.allocate();
    memory.deviceBiclusterScores.copyTo(biclusterScores);
    for (int i = 0; i < memory.InvocationsPerBicluster; i++)
    {
        EXPECT_NEAR(goldMaxScores.data[i], biclusterScores.data[i], 1e-10);
    }
}
