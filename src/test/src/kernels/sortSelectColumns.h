#pragma once

#include <fstream>

#include "../../../LasInstanceMemory.h"
#include "../../../kernels/kernels.h"
#include <gtest/gtest.h>    

class SortSelectColumns : public ::testing::Test
{
protected:

    SortSelectColumns()
            : NumberOfColumns(loadNumberOfColumns()),
              dummyMatrixData(new double[NumberOfColumns]),
              memory(dummyMatrixData, NumberOfColumns, 1, 10),
              goldColumnSet(memory.InvocationsPerBicluster, memory.Width,
                      MatrixLocation::Host),
              goldChanges(memory.InvocationsPerBicluster, 1,
                      MatrixLocation::Host)
    {
        memory.init();
        goldColumnSet.allocate();
        goldChanges.allocate();

        checkCudaErrors(
                cudaMemset(memory.deviceRowChanges.begin(), 1,
                        memory.deviceRowChanges.size()));
        checkCudaErrors(
                cudaMemset(memory.deviceColumnChanges.begin(), 0,
                        memory.deviceColumnChanges.size()));
        checkCudaErrors(
                cudaMemset(memory.deviceColumnSet.begin(), 0,
                        memory.deviceColumnSet.size()));

        std::string tmp;
        std::ifstream infile("data/sortSelectColumns.sums.csv");
        assert(infile.is_open());
        for (int i = 0; i < 5; i++)
            std::getline(infile, tmp);
        for (int i = 0; i < memory.InvocationsPerBicluster; i++)
        {
            for (int j = 0; j < NumberOfColumns; j++)
            {
                double value;
                infile >> value;
                memory.sums.begin()[j * memory.InvocationsPerBicluster + i] =
                        value;
            }
        }
        infile.close();
        memory.sums.copyTo(memory.deviceSums);

        infile.open("data/sortSelectColumns.sizes.csv");
        assert(infile.is_open());
        assert(memory.sizes.width == memory.InvocationsPerBicluster);
        assert(memory.sizes.height == 2);
        for (int i = 0; i < 5; i++)
            std::getline(infile, tmp);
        for (int i = 0; i < memory.InvocationsPerBicluster; i++)
        {
            uint16_t width, height;
            infile >> width >> height;
            memory.sizes.begin()[i * 2] = width;
            memory.sizes.begin()[i * 2 + 1] = height;
        }
        infile.close();
        memory.sizes.copyTo(memory.deviceSizes);

        infile.open("data/sortSelectColumns.goldColumnSet.csv");
        assert(infile.is_open());
        for (int i = 0; i < 5; i++)
            std::getline(infile, tmp);
        for (int i = 0; i < NumberOfColumns; i++)
        {
            for (int j = 0; j < memory.InvocationsPerBicluster; j++)
            {
                double value;
                infile >> value;
                goldColumnSet.begin()[j * memory.columnSet.height + i] = value;
            }
        }
        infile.close();

        infile.open("data/sortSelectColumns.goldChanges.csv");
        assert(infile.is_open());
        for (int i = 0; i < 5; i++)
            std::getline(infile, tmp);
        for (int i = 0; i < memory.InvocationsPerBicluster; i++)
        {
            uint32_t value;
            infile >> value;
            goldChanges.begin()[i] = value;
        }
        infile.close();

    }

    virtual ~SortSelectColumns()
    {
    }

    static uint32_t loadNumberOfColumns()
    {
        std::ifstream infile("data/sortSelectColumns.goldColumnSet.csv");
        assert(infile.is_open());
        std::string tmp;
        for (int i = 0; i < 4; i++)
            std::getline(infile, tmp);

        return std::stoi(tmp.substr(tmp.rfind(' ')));
    }

    const uint16_t NumberOfColumns;
    double* dummyMatrixData;

    LasInstanceMemory memory;

    Matrix<double> goldColumnSet;
    Matrix<uint32_t> goldChanges;

};

TEST_F(SortSelectColumns, Works)
{
    sortSelectColumns(memory);
    //checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    memory.deviceColumnSet.copyTo(memory.columnSet);

    for (int j = 0; j < memory.deviceColumnSet.width; j++)
    {
        for (int i = 0; i < memory.deviceColumnSet.height; i++)
        {
            EXPECT_EQ(goldColumnSet.begin()[j * memory.deviceColumnSet.height + i],
                    memory.columnSet.begin()[j * memory.deviceColumnSet.height + i]);
        }
    }

    Matrix<uint32_t> changes(memory.deviceColumnChanges.width,
            memory.deviceColumnChanges.height, MatrixLocation::Host);
    changes.allocate();
    memory.deviceColumnChanges.copyTo(changes);
    for (int i = 0; i < memory.InvocationsPerBicluster; i++)
    {
        EXPECT_EQ(goldChanges.begin()[i], changes.begin()[i]);
    }

}
