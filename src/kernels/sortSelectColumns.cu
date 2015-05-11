#include "kernels.h"

#include "utilities.cuh"

// Sorts column sums and selects new column set
__global__ void sortSelectColumns_kernel(const double *Sums, double *ColumnSet,
                                         const uint16_t* Sizes,
                                         uint32_t *ColumnChanges, uint16_t MatrixWidth,
                                         uint16_t NextPowerOfTwoAfterWidth)
{
    // columns sums
    extern __shared__ double sums[];
    // columns permutation after sorting
    uint16_t* permutation = (uint16_t*) &sums[NextPowerOfTwoAfterWidth];
    const uint16_t GlobalID = blockIdx.x;
    uint16_t biclusterWidth = Sizes[GlobalID * 2 + 0];


    loadColumnsSums(Sums, sums, permutation, MatrixWidth,
            NextPowerOfTwoAfterWidth);

    sortSums(sums, permutation, NextPowerOfTwoAfterWidth);

    __syncthreads();

    // updating column sets and writing changes
    double flag = 1.0 * int(threadIdx.x < biclusterWidth);
    uint32_t index = GlobalID * MatrixWidth + permutation[threadIdx.x];
    uint8_t changed = uint8_t(flag != ColumnSet[index]);
    ColumnSet[index] = flag;

    index = threadIdx.x + NextPowerOfTwoAfterWidth / 2;
    if (index < MatrixWidth)
    {
        flag = 1.0 * int(index < biclusterWidth);

        index = GlobalID * MatrixWidth + permutation[index];
        changed += uint8_t(flag != ColumnSet[index]);
        ColumnSet[index] = flag;
    }

    atomicAdd(&ColumnChanges[GlobalID], changed);
}

void sortSelectColumns(LasInstanceMemory& memory, uint32_t activeInvocations)
{
    sortSelectColumns_kernel<<< activeInvocations, memory.NextPowerOfTwoAfterWidth / 2,
    sizeof(double) * memory.NextPowerOfTwoAfterWidth +
    sizeof(uint16_t) * memory.NextPowerOfTwoAfterWidth>>>(
            memory.deviceSums.begin(),
            memory.deviceColumnSet.begin(),
            memory.deviceSizes.begin(),
            memory.deviceColumnChanges.begin(),
            memory.Width,
            memory.NextPowerOfTwoAfterWidth);
}
