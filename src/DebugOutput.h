#pragma once

#include <iostream>
#include <cmath>

#include "Matrix.h"

class DebugOutput
{
public:
    DebugOutput() = delete;
    virtual ~DebugOutput() = delete;

    static void log(const std::string& message);

    template<typename T>
    static void printColumnMajorMatrix(Matrix<T>& matrix, uint16_t width,
                                       uint16_t height, bool fixed = true);

    template<typename T>
    static void printColumnMajorMatrix(Matrix<T>& matrix, bool fixed = true);
};

template<typename T>
void DebugOutput::printColumnMajorMatrix(Matrix<T>& matrix, bool fixed)
{
    DebugOutput::printColumnMajorMatrix(matrix, matrix.width, matrix.height,
            fixed);
}

template<typename T>
void DebugOutput::printColumnMajorMatrix(Matrix<T>& matrix, uint16_t width,
                                         uint16_t height, bool fixed)
{
    assert(width * height <= matrix.elements());

    if (fixed) std::cout << std::fixed;

    for (uint32_t i = 0; i < height; i++)
    {
        for (uint32_t j = 0; j < width; j++)
        {
            std::cout << "  " << (matrix.data[j * height + i] >= 0 ? ' ' : '-')
                    << std::abs(matrix.data[j * height + i]);
        }
        std::cout << std::endl;
    }

    if (fixed) std::cout.unsetf(std::ios_base::floatfield);
}
