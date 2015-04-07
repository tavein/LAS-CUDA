#include "kernels.h"

#include "utilities.cuh"

// Sorts row sums and selects new best bicluster height and row set
__global__ void sortSelectHeightAndRows_kernel(
        DeviceMemory memory, const uint16_t MatrixWidth, const uint16_t MatrixHeight,
        const uint16_t NextPowerOfTwoAfterHeight)
{
    extern __shared__ double sums[];
    const uint16_t GlobalID = blockIdx.x;
    uint16_t* permutation = (uint16_t*) &sums[NextPowerOfTwoAfterHeight];
    uint16_t biclusterWidth = memory.Sizes[GlobalID * 2];
    double cncols = memory.gammaln[MatrixWidth + 1] - memory.gammaln[biclusterWidth + 1]
            - memory.gammaln[MatrixWidth - biclusterWidth + 1];
    bool columnsChanged = memory.ColumnChanges[GlobalID] > 0;

    if (columnsChanged)
    {

        loadRowsSums(memory.Sums, sums, permutation, MatrixHeight,
                NextPowerOfTwoAfterHeight);

        sortSums(sums, permutation, NextPowerOfTwoAfterHeight);

        __syncthreads();

        const uint16_t LocalID1 = 2 * threadIdx.x;
        const uint16_t LocalID2 = LocalID1 + 1;
        cumsum(sums, NextPowerOfTwoAfterHeight, LocalID1, LocalID2);

        double score1 = LasScoreHeight(memory.gammaln, sums[LocalID1], MatrixHeight, biclusterWidth, LocalID1 + 1, cncols);
        double score2 = LasScoreHeight(memory.gammaln, sums[LocalID2], MatrixHeight, biclusterWidth, LocalID2 + 1, cncols);


        const uint16_t rowId1 = permutation[threadIdx.x];
        const uint16_t rowId2 = permutation[threadIdx.x
                + NextPowerOfTwoAfterHeight / 2];

        __syncthreads();

        selectMaxScoreDimension(LocalID1, LocalID2,
                                sums, permutation,
                                MatrixHeight, NextPowerOfTwoAfterHeight,
                                score1, score2);

        // writing new size and bicluster score
        if (LocalID1 == 0)
        {
            memory.Sizes[2 * GlobalID + 1] = permutation[0] + 1;
            memory.Scores[GlobalID] = sums[0];
        }

        // writing new row set and number of changed rows
        double flag = 1.0 * int(threadIdx.x <= permutation[0]);
        uint32_t index = rowId1 * gridDim.x + GlobalID;
        uint8_t changed = uint8_t(memory.RowSet[index] != flag);
        memory.RowSet[index] = flag;

        index = threadIdx.x + NextPowerOfTwoAfterHeight / 2;
        if (index < MatrixHeight)
        {
            flag = 1.0 * int(index <= permutation[0]);

            index = rowId2 * gridDim.x + GlobalID;
            changed += uint8_t(memory.RowSet[index] != flag);
            memory.RowSet[index] = flag;
        }

        atomicAdd(&memory.RowChanges[GlobalID], changed);
    }
}

void sortSelectHeightAndRows(LasInstanceMemory& memory)
{
    sortSelectHeightAndRows_kernel<<< memory.InvocationsPerBicluster, memory.NextPowerOfTwoAfterHeight/2,
    sizeof(double) * memory.NextPowerOfTwoAfterHeight +
    sizeof(uint16_t) * memory.NextPowerOfTwoAfterHeight>>>(
            memory.deviceMemory,
            memory.Width, memory.Height,
            memory.NextPowerOfTwoAfterHeight);

}
