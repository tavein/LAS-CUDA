#pragma once

#include <fstream>

#include "../../../LasInstanceMemory.h"
#include "../../../kernels/kernels.h"
#include <gtest/gtest.h>    

class SortSelectRows : public ::testing::Test
{
protected:

    SortSelectRows()
            : NumberOfRows(loadNumberOfRows()),
              dummyMatrixData(new double[NumberOfRows]),
              memory(dummyMatrixData, 1, NumberOfRows, 10),
              goldRowSet(memory.Height, memory.InvocationsPerBicluster,
                      MatrixLocation::Host),
              goldChanges(memory.InvocationsPerBicluster, 1,
                      MatrixLocation::Host)
    {

        memory.init();
        goldRowSet.allocate();
        goldChanges.allocate();
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

        std::ifstream infile("data/sortSelectRows.sums.csv");
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

        infile.open("data/sortSelectRows.sizes.csv");
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

        infile.open("data/sortSelectRows.goldRowSet.csv");
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

        infile.open("data/sortSelectRows.goldChanges.csv");
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

    }

    virtual ~SortSelectRows()
    {
    }

    static uint32_t loadNumberOfRows()
    {
        std::ifstream infile("data/sortSelectRows.goldRowSet.csv");
        assert(infile.is_open());
        std::string tmp;
        for (int i = 0; i < 5; i++)
            std::getline(infile, tmp);

        return std::stoi(tmp.substr(tmp.rfind(' ')));
    }

    const uint16_t NumberOfRows;
    double* dummyMatrixData;

    LasInstanceMemory memory;

    Matrix<double> goldRowSet;
    Matrix<uint32_t> goldChanges;
};

TEST_F(SortSelectRows, Works)
{
    sortSelectRows(memory);
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
}
