#include "kernels.h"

#include "utilities.cuh"

// Sorts column sums and selects new best bicluster width and column set
__global__ void sortSelectWidthAndColumns_kernel(
        DeviceMemory memory, const uint16_t MatrixHeight, const uint16_t MatrixWidth,
        const uint16_t NextPowerOfTwoAfterWidth)
{
    extern __shared__ double sums[];
    __shared__ uint16_t* permutation;
    __shared__ uint16_t biclusterHeight;
    __shared__ double cnrows;
    __shared__ bool rowsChanged;
    const uint16_t GlobalID = blockIdx.x;
    if (threadIdx.x == 0)
    {
        permutation = (uint16_t*) &sums[NextPowerOfTwoAfterWidth];
        biclusterHeight = memory.Sizes[GlobalID * 2 + 1];
        cnrows = memory.gammaln[MatrixHeight + 1] - memory.gammaln[biclusterHeight + 1]
                - memory.gammaln[MatrixHeight - biclusterHeight + 1];
        rowsChanged = memory.RowChanges[GlobalID] > 0;
    }

    __syncthreads();

    if (rowsChanged)
    {

        loadColumnsSums(memory.Sums, sums, permutation, MatrixWidth,
                NextPowerOfTwoAfterWidth);

        sortSums(sums, permutation, NextPowerOfTwoAfterWidth);

        __syncthreads();

        const uint16_t LocalID1 = 2 * threadIdx.x;
        const uint16_t LocalID2 = LocalID1 + 1;
        cumsum(sums, NextPowerOfTwoAfterWidth, LocalID1, LocalID2);

        double score1 = LasScoreWidth(memory.gammaln, sums[LocalID1], MatrixWidth, LocalID1 + 1, biclusterHeight, cnrows);
        double score2 = LasScoreWidth(memory.gammaln, sums[LocalID2], MatrixWidth, LocalID2 + 1, biclusterHeight, cnrows);

        const uint16_t colId1 = permutation[threadIdx.x];
        const uint16_t colId2 = permutation[threadIdx.x
                + NextPowerOfTwoAfterWidth / 2];

        __syncthreads();

        selectMaxScoreDimension(LocalID1, LocalID2,
                                sums, permutation,
                                MatrixWidth, NextPowerOfTwoAfterWidth,
                                score1, score2);

        // writing new size and bicluster score
        if (LocalID1 == 0)
        {
            memory.Sizes[2 * GlobalID] = permutation[0] + 1;
            memory.Scores[GlobalID] = sums[0];
        }

        // writing new column set and number of changed columns
        double flag = 1.0 * int(threadIdx.x <= permutation[0]);
        uint32_t index = GlobalID * MatrixWidth + colId1;
        uint8_t changed = uint8_t(memory.ColumnSet[index] != flag);
        memory.ColumnSet[index] = flag;

        index = threadIdx.x + NextPowerOfTwoAfterWidth / 2;
        if (index < MatrixWidth)
        {
            flag = 1.0 * int(index <= permutation[0]);

            index = GlobalID * MatrixWidth + colId2;
            changed += uint8_t(memory.ColumnSet[index] != flag);
            memory.ColumnSet[index] = flag;
        }

        atomicAdd(&memory.ColumnChanges[GlobalID], changed);
    }
}

void sortSelectWidthAndColumns(LasInstanceMemory& memory)
{
    sortSelectWidthAndColumns_kernel<<< memory.InvocationsPerBicluster, memory.NextPowerOfTwoAfterWidth/2,
    sizeof(double) * memory.NextPowerOfTwoAfterWidth +
    sizeof(uint16_t) * memory.NextPowerOfTwoAfterWidth>>>(
            memory.deviceMemory,
            memory.Height, memory.Width,
            memory.NextPowerOfTwoAfterWidth);

}
