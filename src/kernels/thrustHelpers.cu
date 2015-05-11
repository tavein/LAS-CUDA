#include "kernels.h"

#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/extrema.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>

// computes index and value of score maximum
std::pair<uint32_t, double> maxScore(Matrix<double>& deviceScores)
{
    thrust::device_ptr<double> deviceScores_ptr(deviceScores.begin());

    thrust::device_ptr<double> max_ptr = thrust::max_element(deviceScores_ptr,
            deviceScores_ptr + deviceScores.elements());
    return std::make_pair(max_ptr - deviceScores_ptr, *max_ptr);
}

// sorts changes in descending order and generates sorting permutation
uint32_t sortSelectActiveInvocations(Matrix<uint32_t>& changes, Matrix<uint16_t>& permutation,
                                    uint32_t activeInvocations) {

    thrust::device_ptr<uint16_t> permutation_ptr(permutation.begin());
    thrust::sequence(permutation_ptr, permutation_ptr + activeInvocations + (activeInvocations % 2));

    thrust::device_ptr<uint32_t> changes_ptr(changes.begin());

    thrust::sort_by_key(changes_ptr, changes_ptr + activeInvocations, permutation_ptr, thrust::greater<uint32_t>());

    return (thrust::lower_bound(changes_ptr, changes_ptr + activeInvocations, 0, thrust::greater<uint32_t>()) - changes_ptr);
}
