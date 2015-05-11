#include "kernels.h"

#include "utilities.cuh"

__global__ void reorderSizesAndScores_kernel(
        uint16_t* Sizes,
        double* Scores,
        const uint16_t* Permutation)
{
    extern __shared__ uint16_t sizes[];
    double* scores = (double*)sizes;

    const uint16_t LocalID1 = threadIdx.x*2 + 0;
    const uint16_t LocalID2 = threadIdx.x*2 + 1;

    sizes[LocalID1*2 + 0] = Sizes[LocalID1*2 + 0];
    sizes[LocalID1*2 + 1] = Sizes[LocalID1*2 + 1];

    sizes[LocalID2*2 + 0] = Sizes[LocalID2*2 + 0];
    sizes[LocalID2*2 + 1] = Sizes[LocalID2*2 + 1];

    __syncthreads();

    const uint16_t NewLocalID1 = Permutation[LocalID1];
    Sizes[LocalID1*2 + 0] = sizes[NewLocalID1*2 + 0];
    Sizes[LocalID1*2 + 1] = sizes[NewLocalID1*2 + 1];

    const uint16_t NewLocalID2 = Permutation[LocalID2];
    Sizes[LocalID2*2 + 0] = sizes[NewLocalID2*2 + 0];
    Sizes[LocalID2*2 + 1] = sizes[NewLocalID2*2 + 1];

    __syncthreads();

    scores[LocalID1] = Scores[LocalID1];
    scores[LocalID2] = Scores[LocalID2];

    __syncthreads();

    Scores[LocalID1] = scores[NewLocalID1];
    Scores[LocalID2] = scores[NewLocalID2];
}

void reorderSizesAndScores(LasInstanceMemory& memory, uint32_t activeInvocationsPerBicluster)
{
    activeInvocationsPerBicluster += activeInvocationsPerBicluster % 2;

    reorderSizesAndScores_kernel<<< 1, activeInvocationsPerBicluster / 2,
            std::max(sizeof(uint16_t) * 2 * activeInvocationsPerBicluster,
                     sizeof(double) * activeInvocationsPerBicluster)>>>(
            memory.deviceSizes.data,
            memory.deviceBiclusterScores.data,
            memory.deviceInvocationsPermutation.data);
}
