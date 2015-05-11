#include "kernels.h"

#include "utilities.cuh"

__global__ void reorderRowSets_kernel(
        double *RowSet,
        const uint16_t* Permutation,
        uint16_t InvocationsCount)
{
    extern __shared__ double sums[];

    const uint16_t GlobalID = blockIdx.x;
    const uint16_t LocalID1 = threadIdx.x*2 + 0;
    const uint16_t LocalID2 = threadIdx.x*2 + 1;

    sums[LocalID1] = RowSet[GlobalID * InvocationsCount + LocalID1];
    sums[LocalID2] = RowSet[GlobalID * InvocationsCount + LocalID2];

    __syncthreads();

    RowSet[GlobalID * InvocationsCount + LocalID1] = sums[Permutation[LocalID1]];
    RowSet[GlobalID * InvocationsCount + LocalID2] = sums[Permutation[LocalID2]];
}

void reorderRowSets(LasInstanceMemory& memory, uint32_t activeInvocationsPerBicluster)
{
    activeInvocationsPerBicluster += activeInvocationsPerBicluster % 2;

    reorderRowSets_kernel<<< memory.Height, activeInvocationsPerBicluster / 2,
            sizeof(double) * activeInvocationsPerBicluster>>>(
            memory.deviceRowSet.data,
            memory.deviceInvocationsPermutation.data,
            memory.InvocationsPerBicluster);
}
