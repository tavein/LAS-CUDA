#pragma once

#include <iostream>
#include <cassert>
#include <stdint.h>
#include <iso646.h>

#include "ErrorHandling.h"

enum MatrixLocation
{
    Device, Host
};

// Utility class for data passing to and from device
template<typename T>
class Matrix
{
public:
    Matrix(const uint32_t& width, const uint32_t& height,
           const MatrixLocation& location);
    virtual ~Matrix();

    void allocate();
    void release();

    Matrix<T>& operator =(const Matrix<T>& otherMatrix) = delete;
    Matrix(const Matrix<T>& otherMatrix) = delete;

    void copyTo(Matrix<T>& otherMatrix);
    void copyFrom(Matrix<T>& otherMatrix);

    std::size_t size() const;
    uint32_t elements() const;
    T* begin();
    T* end();

    const uint32_t width;
    const uint32_t height;

    const MatrixLocation location;

    T* data;
};

template<typename T>
Matrix<T>::Matrix(const uint32_t& width, const uint32_t& height, const MatrixLocation& location)
: width(width)
, height(height)
, location(location)
, data(nullptr)
{

}

template<typename T>
void Matrix<T>::allocate()
{
    if (data == nullptr)
    {
        if (location == Host)
        {
            data = new T[width * height];
        }
        else
        {
            checkCudaErrors(cudaMalloc(&data, sizeof(T) * width * height));
        assert(data != nullptr);
    }
}
}

template<typename T>
void Matrix<T>::release()
{
if (data != nullptr)
{
    if (location == Host)
    {
        delete[] data;
    }
    else
    {
        checkCudaErrors(cudaFree(data));
    }
data = nullptr;
}
}

template<typename T>
Matrix<T>::~Matrix()
{
release();
}

template<typename T>
void Matrix<T>::copyTo(Matrix<T>& otherMatrix)
{
assert(this->data != nullptr and otherMatrix.data != nullptr);

    assert(size() <= otherMatrix.size());
if (width != otherMatrix.width or height != otherMatrix.height)
{
std::cerr << "Copying form matrix of size " << width << ", " << height << ") "
        << "to matrix of size " << otherMatrix.width << ", "
        << otherMatrix.height << "). " << "Hope you know what you are doing."
        << std::endl;
}
cudaMemcpyKind memcpyKind;
if (otherMatrix.location == Device and this->location == Device)
{
memcpyKind = cudaMemcpyDeviceToDevice;
}
else if (otherMatrix.location == Host and this->location == Host)
{
memcpyKind = cudaMemcpyHostToHost;
}
else if (otherMatrix.location == Host)
{
memcpyKind = cudaMemcpyDeviceToHost;
}
else
{
memcpyKind = cudaMemcpyHostToDevice;
}

checkCudaErrors(cudaMemcpy(otherMatrix.data, this->data, size(), memcpyKind));
}

template<typename T>
void Matrix<T>::copyFrom(Matrix<T>& otherMatrix)
{
otherMatrix.copyTo(*this);
}

template<typename T>
uint32_t Matrix<T>::elements() const
{
return width * height;
}

template<typename T>
std::size_t Matrix<T>::size() const
{
return sizeof(T) * elements();
}

template<typename T>
T* Matrix<T>::begin()
{
assert(data != nullptr);

return data;
}

template<typename T>
T* Matrix<T>::end()
{
return data + elements();
}
