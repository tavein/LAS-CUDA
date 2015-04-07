#pragma once

#include "LasException.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

const char* _cudaGetErrorString(cudaError_t error);

const char *_cudaGetErrorString(cublasStatus_t error);

template<typename T>
void check(T result, char const * const func, const char * const file,
           int const line)
{
    if (result != 0)
    {
        std::string functionCall = func;
        functionCall = functionCall.substr(0, functionCall.find('('));

        std::string message = "CUDA error #" + std::to_string(result)
                + " in file " + file + ", at line " + std::to_string(line)
                + ", when calling " + functionCall + ": "
                + _cudaGetErrorString(result);

        throw LasException(message);
    }
}
#define checkCudaErrors(val)           check ( (val), #val, __FILE__, __LINE__ )

