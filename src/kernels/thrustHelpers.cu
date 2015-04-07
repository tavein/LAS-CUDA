#include "kernels.h"

#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/extrema.h>

// computes total number of changes in sets
uint32_t reduceChanges(Matrix<uint32_t>& deviceChanges)
{
    thrust::device_ptr<uint32_t> deviceColumnChanges_ptr(deviceChanges.begin());

    return thrust::reduce(deviceColumnChanges_ptr,
            deviceColumnChanges_ptr + deviceChanges.elements());
}

// computes index and value of score maximum
std::pair<uint32_t, double> maxScore(Matrix<double>& deviceScores)
{
    thrust::device_ptr<double> deviceScores_ptr(deviceScores.begin());

    thrust::device_ptr<double> max_ptr = thrust::max_element(deviceScores_ptr,
            deviceScores_ptr + deviceScores.elements());
    return std::make_pair(max_ptr - deviceScores_ptr, *max_ptr);
}
