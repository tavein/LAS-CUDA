#pragma once

#include <fstream>

#include "../../../LasInstanceMemory.h"
#include "../../../kernels/kernels.h"
#include <gtest/gtest.h>    

class SortSelectHeightAndRows : public ::testing::Test
{
protected:

    SortSelectHeightAndRows()
            : MatrixSize(loadMatrixSize()),
              dummyMatrixData(new double[MatrixSize.first * MatrixSize.second]),
              memory(dummyMatrixData, MatrixSize.first, MatrixSize.second, 10),
              goldRowSet(memory.rowSet.width, memory.rowSet.height,
                      MatrixLocation::Host),
              goldChanges(memory.deviceRowChanges.width,
                      memory.deviceRowChanges.height, MatrixLocation::Host),
              goldNewSizes(memory.sizes.width, memory.sizes.height,
                      MatrixLocation::Host)
    {
        memory.init();
        goldRowSet.allocate();
        goldChanges.allocate();
        goldNewSizes.allocate();
        std::string tmp;

        checkCudaErrors(
                cudaMemset(memory.deviceColumnChanges.begin(), 1,
                        memory.deviceColumnChanges.size()));
        checkCudaErrors(
                cudaMemset(memory.deviceRowChanges.begin(), 0,
                        memory.deviceRowChanges.size()));
        checkCudaErrors(
                cudaMemset(memory.deviceRowSet.data, 0,
                        memory.deviceRowSet.size()));

        std::ifstream infile("data/sortSelectHeightAndRows.sums.csv");
        assert(infile.is_open());
        for (int i = 0; i < 5; i++)
            std::getline(infile, tmp);
        for (int i = 0; i < memory.Height; i++)
        {
            for (int j = 0; j < memory.InvocationsPerBicluster; j++)
            {
                double value;
                infile >> value;
                memory.sums.data[j * memory.Height + i] = value;
            }
        }
        infile.close();
        memory.sums.copyTo(memory.deviceSums);

        infile.open("data/sortSelectHeightAndRows.sizes.csv");
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

        infile.open("data/sortSelectHeightAndRows.goldRowSet.csv");
        assert(infile.is_open());
        for (int i = 0; i < 5; i++)
            std::getline(infile, tmp);
        for (int i = 0; i < memory.rowSet.height; i++)
        {
            for (int j = 0; j < memory.rowSet.width; j++)
            {
                double value;
                infile >> value;
                goldRowSet.data[j * memory.rowSet.height + i] = value;
            }
        }
        infile.close();

        infile.open("data/sortSelectHeightAndRows.goldChanges.csv");
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

        infile.open("data/sortSelectHeightAndRows.goldNewSizes.csv");
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
    }

    virtual ~SortSelectHeightAndRows()
    {
    }

    static std::pair<uint16_t, uint16_t> loadMatrixSize()
    {
        uint32_t M, N;

        std::ifstream infile("data/sortSelectHeightAndRows.matrixSize.csv");
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

    Matrix<double> goldRowSet;
    Matrix<uint32_t> goldChanges;
    Matrix<uint16_t> goldNewSizes;
};

TEST_F(SortSelectHeightAndRows, Works)
{
    sortSelectHeightAndRows(this->memory);
    checkCudaErrors(cudaGetLastError());

    memory.deviceRowSet.copyTo(memory.rowSet);
    for (int j = 0; j < memory.deviceRowSet.width; j++)
    {
        for (int i = 0; i < memory.deviceRowSet.height; i++)
        {
            EXPECT_EQ(goldRowSet.data[j * memory.deviceRowSet.height + i],
                    memory.rowSet.data[j * memory.deviceRowSet.height + i]);
        }
    }

    Matrix<uint32_t> changes(memory.deviceRowChanges.width,
            memory.deviceRowChanges.height, MatrixLocation::Host);
    changes.allocate();
    memory.deviceRowChanges.copyTo(changes);
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
}
