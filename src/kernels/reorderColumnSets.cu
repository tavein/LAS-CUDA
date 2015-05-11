#include "kernels.h"

#include "utilities.cuh"


__global__ void reorderColumnSets_kernel(
        double *ColumnSet,
        const uint16_t* Permutation,
        uint16_t MatrixWidth)
{
    extern __shared__ double sums[];

    const uint16_t GlobalID = blockIdx.x;
    const uint16_t LocalID1 = threadIdx.x*2 + 0;
    const uint16_t LocalID2 = threadIdx.x*2 + 1;

    sums[LocalID1] = ColumnSet[LocalID1 * MatrixWidth + GlobalID];
    sums[LocalID2] = ColumnSet[LocalID2 * MatrixWidth + GlobalID];

    __syncthreads();

    ColumnSet[LocalID1 * MatrixWidth + GlobalID] = sums[Permutation[LocalID1]];
    ColumnSet[LocalID2 * MatrixWidth + GlobalID] = sums[Permutation[LocalID2]];
}

void reorderColumnSets(LasInstanceMemory& memory, uint32_t activeInvocationsPerBicluster)
{
    activeInvocationsPerBicluster += activeInvocationsPerBicluster % 2;

    reorderColumnSets_kernel<<< memory.Width, activeInvocationsPerBicluster / 2,
            sizeof(double) * activeInvocationsPerBicluster>>>(
            memory.deviceColumnSet.data,
            memory.deviceInvocationsPermutation.data,
            memory.Width);
}
