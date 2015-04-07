/*
 *
 * For the sorting and cumulative sum they asked me to provide this.
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
*/


#pragma once

#include "../Matrix.h"

// bitonic sort, copypasted from CUDA samples and slightly modified

__inline__ __device__ void Comparator(double &keyA, uint16_t &valA,
                                      double &keyB, uint16_t &valB,
                                      bool arrowDir)
{

    if ((keyA > keyB) == arrowDir)
    {
        double tk = keyA;
        keyA = keyB;
        keyB = tk;

        uint16_t tv = valA;
        valA = valB;
        valB = tv;
    }
}

__inline__ __device__ void sortSums(
        double* sums, uint16_t* permutation,
        const uint16_t& NextPowerOfTwoAfterDimension)
{

    for (uint16_t size = 2; size < NextPowerOfTwoAfterDimension; size <<= 1)
    {
        //Bitonic merge
        bool dir = (threadIdx.x & (size / 2)) != 0;

        for (uint16_t stride = size / 2; stride > 0; stride >>= 1)
        {
            __syncthreads();
            uint16_t pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
            Comparator(sums[pos + 0], permutation[pos + 0], sums[pos + stride],
                    permutation[pos + stride], dir);
        }
    }

    {
        for (uint16_t stride = NextPowerOfTwoAfterDimension / 2; stride > 0;
                stride >>= 1)
        {
            __syncthreads();
            uint16_t pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
            Comparator(sums[pos + 0], permutation[pos + 0], sums[pos + stride],
                    permutation[pos + stride], false);
        }
    }

}

__inline__ __device__ void loadRowsSums(
        const double* Sums, double* sharedSums, uint16_t* permutation,
        const uint16_t& Height, const uint16_t& NextPowerOfTwoAfterHeight)
{

    sharedSums[threadIdx.x + 0] = Sums[blockIdx.x * Height + threadIdx.x];
    permutation[threadIdx.x + 0] = threadIdx.x;

    uint32_t index = threadIdx.x + NextPowerOfTwoAfterHeight / 2;
    if (index < Height)
    {
        sharedSums[index] = Sums[blockIdx.x * Height + index];
        permutation[index] = index;
    }
    else
    {
        sharedSums[index] = -INFINITY;
    }
}

__inline__ __device__ void loadColumnsSums(
        const double* Sums, double* sharedSums, uint16_t* permutation,
        const uint16_t& Width, const uint16_t& NextPowerOfTwoAfterWidth)
{

    sharedSums[threadIdx.x + 0] = Sums[blockIdx.x + threadIdx.x * gridDim.x];
    permutation[threadIdx.x + 0] = threadIdx.x;

    uint32_t index = threadIdx.x + NextPowerOfTwoAfterWidth / 2;
    if (index < Width)
    {
        sharedSums[index] = Sums[blockIdx.x + index * gridDim.x];
        permutation[index] = index;
    }
    else
    {
        sharedSums[index] = -INFINITY;
    }
}

// computes cumulative sum (also stolen from CUDA samples)
__inline__ __device__ void cumsum(double* sums,
                                  const uint16_t& NextPowerOfTwoAfterDimension,
                                  const uint16_t& LocalID1,
                                  const uint16_t& LocalID2)
{
    for (uint16_t i = 1; i < NextPowerOfTwoAfterDimension; i <<= 1)
    {
        int flag = int(LocalID1 >= i);
        double v1 = flag * sums[(LocalID1 - i) * flag];
        flag = int(LocalID2 >= i);
        double v2 = flag * sums[(LocalID2 - i) * flag];
        __syncthreads();

        sums[LocalID1] += v1;
        sums[LocalID2] += v2;
        __syncthreads();
    }
}

// LAS score for fixed bicluster width and variable height
__inline__ __device__ double LasScoreHeight(double* gammaln, double cumsum,
                                            uint16_t MatrixHeight,
                                            uint16_t biclusterWidth,
                                            uint16_t biclusterHeight,
                                            double cncols)
{
    double cnrows = gammaln[MatrixHeight + 1] - gammaln[biclusterHeight + 1]
            - gammaln[MatrixHeight - biclusterHeight + 1];
    double ar = cumsum / sqrt(double(biclusterHeight) * double(biclusterWidth));
    double rest2 = -ar * ar / 2 + log(erfcx(ar / sqrt(2.0)) / 2.0);
    return -rest2 - cnrows - cncols;
}

// LAS score for fixed bicluster height and variable width
__inline__ __device__ double LasScoreWidth(double* gammaln, double cumsum,
                                           uint16_t MatrixWidth,
                                           uint16_t biclusterWidth,
                                           uint16_t biclusterHeight,
                                           double cnrows)
{
    double cncols = gammaln[MatrixWidth + 1] - gammaln[biclusterWidth + 1]
            - gammaln[MatrixWidth - biclusterWidth + 1];
    double ar = cumsum / sqrt(double(biclusterHeight) * double(biclusterWidth));
    double rest2 = -ar * ar / 2 + log(erfcx(ar / sqrt(2.0)) / 2.0);
    return -rest2 - cnrows - cncols;
}

__inline__ __device__ void selectMaxScoreDimension(
        uint16_t LocalID1, uint16_t LocalID2, double* sums,
        uint16_t* permutation, uint16_t Dimension,
        uint16_t NextPowerOfTwoAfterDimension, double score1, double score2)
{
    {
        bool condition = score1 >= score2;
        sums[LocalID1] = condition ? score1 : score2;
        permutation[LocalID1] = condition ? LocalID1 : LocalID2;
    }

    __syncthreads();

    for (uint16_t d = 2; d < Dimension; d <<= 1)
    {
        uint16_t k = LocalID1 + d * uint16_t(LocalID1 + d < Dimension);
        double value = max(sums[LocalID1], sums[k]);
        uint16_t biclusterDimension =
                sums[LocalID1] >= sums[k] ?
                        permutation[LocalID1] : permutation[k];

        __syncthreads();

        sums[LocalID1] = value;
        permutation[LocalID1] = biclusterDimension;

        __syncthreads();
    }
}
