#include "kernels.h"

#include "utilities.cuh"

// Sorts row sums and selects new row set
__global__ void sortSelectRows_kernel(const double *Sums, double *RowSet,
                                      const uint16_t* Sizes,
                                      const uint32_t* ColumnChanges,
                                      uint32_t *RowChanges, uint16_t MatrixHeight,
                                      uint16_t NextPowerOfTwoAfterHeight)
{
    extern __shared__ double sums[];
    const uint16_t GlobalID = blockIdx.x;
    uint16_t* permutation = (uint16_t*) &sums[NextPowerOfTwoAfterHeight];
    uint16_t biclusterHeight = Sizes[GlobalID * 2 + 1];
    bool columnsChanged = ColumnChanges[GlobalID] > 0;

    if (columnsChanged)
    {

        loadRowsSums(Sums, sums, permutation, MatrixHeight,
                NextPowerOfTwoAfterHeight);

        sortSums(sums, permutation, NextPowerOfTwoAfterHeight);

        __syncthreads();

        // writing new row set and updating count of changed elements in it
        double flag = 1.0 * int(threadIdx.x < biclusterHeight);
        uint32_t index = permutation[threadIdx.x] * gridDim.x + GlobalID;
        uint8_t changed = uint8_t(RowSet[index] != flag);
        RowSet[index] = flag;

        index = threadIdx.x + NextPowerOfTwoAfterHeight / 2;
        if (index < MatrixHeight)
        {
            flag = 1.0 * int(index < biclusterHeight);

            index = permutation[index] * gridDim.x + GlobalID;
            changed += uint8_t(RowSet[index] != flag);
            RowSet[index] = flag;
        }

        atomicAdd(&RowChanges[GlobalID], changed);
    }
}

void sortSelectRows(LasInstanceMemory& memory)
{
    sortSelectRows_kernel<<< memory.InvocationsPerBicluster, memory.NextPowerOfTwoAfterHeight / 2,
    sizeof(double) * memory.NextPowerOfTwoAfterHeight +
    sizeof(uint16_t) * memory.NextPowerOfTwoAfterHeight >>>(
            memory.deviceSums.begin(),
            memory.deviceRowSet.begin(),
            memory.deviceSizes.begin(),
            memory.deviceColumnChanges.begin(),
            memory.deviceRowChanges.begin(),
            memory.Height,
            memory.NextPowerOfTwoAfterHeight);

}
