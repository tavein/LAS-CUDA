#pragma once

#include <fstream>
#include <algorithm>

#include "../../../LasInstanceMemory.h"
#include "../../../kernels/kernels.h"
#include <gtest/gtest.h>    

class ReorderRowSets : public ::testing::Test
{
protected:

    ReorderRowSets()
            : NumberOfRows(766),
              NumberOfColumns(124),
              dummyMatrixData(new double[NumberOfRows*NumberOfColumns]),
              memory(dummyMatrixData, NumberOfColumns, NumberOfRows, 100),
              activeInvocations(std::uniform_int_distribution<>(memory.InvocationsPerBicluster/2, memory.InvocationsPerBicluster)(MT19937_GENERATOR)),
              originalRowSet(memory.Height, memory.InvocationsPerBicluster,
                      MatrixLocation::Host),
              permutation(memory.InvocationsPerBicluster, 1, MatrixLocation::Host)
    {

        memory.init();
        originalRowSet.allocate();
        permutation.allocate();

        std::iota(permutation.data,
                  permutation.data + memory.deviceInvocationsPermutation.elements(),
                  0);
        std::random_shuffle(permutation.data,
                            permutation.data + activeInvocations);
        permutation.copyTo(memory.deviceInvocationsPermutation);

        for (int i = 0; i < memory.Height; i++)
        {
            for (int j = 0; j < memory.InvocationsPerBicluster; j++)
            {
                memory.rowSet.data[i * memory.InvocationsPerBicluster + j] = UNIFORM_REAL(MT19937_GENERATOR);
            }
        }
        memory.rowSet.copyTo(memory.deviceRowSet);
        memory.rowSet.copyTo(originalRowSet);
    }

    virtual ~ReorderRowSets()
    {
        delete[] dummyMatrixData;
    }

    const uint16_t NumberOfRows;
    const uint16_t NumberOfColumns;
    double* dummyMatrixData;

    LasInstanceMemory memory;
    const uint32_t activeInvocations;

    Matrix<double> originalRowSet;
    Matrix<uint16_t> permutation;
};

TEST_F(ReorderRowSets, Works)
{
    reorderRowSets(memory, activeInvocations);
    checkCudaErrors(cudaGetLastError());

    memory.deviceRowSet.copyTo(memory.rowSet);
    for (int j = 0; j < memory.deviceRowSet.width; j++)
    {
        for (int i = 0; i < activeInvocations; i++)
        {
            EXPECT_EQ(originalRowSet.data[j * memory.deviceRowSet.height + permutation.data[i]],
                       memory.rowSet.data[j * memory.deviceRowSet.height + i]);
        }
    }
}
