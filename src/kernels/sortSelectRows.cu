#include "kernels.h"

#include "utilities.cuh"

// Sorts row sums and selects new row set
__global__ void sortSelectRows_kernel(const double *Sums, double *RowSet,
                                      const uint16_t* Sizes,
                                      uint32_t *RowChanges, uint16_t MatrixHeight,
                                      const uint16_t NextPowerOfTwoAfterHeight,
                                      const uint32_t TotalInvocations)
{
    extern __shared__ double sums[];
    const uint16_t GlobalID = blockIdx.x;
    uint16_t* permutation = (uint16_t*) &sums[NextPowerOfTwoAfterHeight];
    uint16_t biclusterHeight = Sizes[GlobalID * 2 + 1];


    loadRowsSums(Sums, sums, permutation, MatrixHeight,
            NextPowerOfTwoAfterHeight);

    sortSums(sums, permutation, NextPowerOfTwoAfterHeight);

    __syncthreads();

    // writing new row set and updating count of changed elements in it
    double flag = 1.0 * int(threadIdx.x < biclusterHeight);
    uint32_t index = permutation[threadIdx.x] * TotalInvocations + GlobalID;
    uint8_t changed = uint8_t(RowSet[index] != flag);
    RowSet[index] = flag;

    index = threadIdx.x + NextPowerOfTwoAfterHeight / 2;
    if (index < MatrixHeight)
    {
        flag = 1.0 * int(index < biclusterHeight);

        index = permutation[index] * TotalInvocations + GlobalID;
        changed += uint8_t(RowSet[index] != flag);
        RowSet[index] = flag;
    }

    atomicAdd(&RowChanges[GlobalID], changed);
}

void sortSelectRows(LasInstanceMemory& memory, uint32_t activeInvocations)
{
    sortSelectRows_kernel<<< activeInvocations, memory.NextPowerOfTwoAfterHeight / 2,
    sizeof(double) * memory.NextPowerOfTwoAfterHeight +
    sizeof(uint16_t) * memory.NextPowerOfTwoAfterHeight >>>(
            memory.deviceSums.begin(),
            memory.deviceRowSet.begin(),
            memory.deviceSizes.begin(),
            memory.deviceRowChanges.begin(),
            memory.Height,
            memory.NextPowerOfTwoAfterHeight,
            memory.InvocationsPerBicluster);

}
