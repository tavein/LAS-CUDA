#include "kernels.h"

#include "utilities.cuh"

// Sorts column sums and selects new column set
__global__ void sortSelectColumns_kernel(const double *Sums, double *ColumnSet,
                                         const uint16_t* Sizes,
                                         const uint32_t* RowChanges,
                                         uint32_t *ColumnChanges, uint16_t MatrixWidth,
                                         uint16_t NextPowerOfTwoAfterWidth)
{
    // columns sums
    extern __shared__ double sums[];
    // columns permutation after sorting
    __shared__ uint16_t* permutation;
    __shared__ uint16_t biclusterWidth;
    __shared__ bool rowsChanged;
    const uint16_t GlobalID = blockIdx.x;

    if (threadIdx.x == 0)
    {
        permutation = (uint16_t*) &sums[NextPowerOfTwoAfterWidth];
        biclusterWidth = Sizes[GlobalID * 2 + 0];
        rowsChanged = RowChanges[GlobalID] > 0;
    }

    __syncthreads();

    if (rowsChanged)
    {

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
}

void sortSelectColumns(LasInstanceMemory& memory)
{
    sortSelectColumns_kernel<<< memory.InvocationsPerBicluster, memory.NextPowerOfTwoAfterWidth / 2,
    sizeof(double) * memory.NextPowerOfTwoAfterWidth +
    sizeof(uint16_t) * memory.NextPowerOfTwoAfterWidth>>>(
            memory.deviceSums.begin(),
            memory.deviceColumnSet.begin(),
            memory.deviceSizes.begin(),
            memory.deviceRowChanges.begin(),
            memory.deviceColumnChanges.begin(),
            memory.Width,
            memory.NextPowerOfTwoAfterWidth);
}
