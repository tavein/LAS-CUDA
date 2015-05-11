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
                                       uint16_t height, bool fixed = true,
                                       bool transpose = false);

    template<typename T>
    static void printColumnMajorMatrix(Matrix<T>& matrix, bool fixed = true,
                                       bool transpose = false);
};

template<typename T>
void DebugOutput::printColumnMajorMatrix(Matrix<T>& matrix, bool fixed, bool transpose)
{
    DebugOutput::printColumnMajorMatrix(matrix, matrix.width, matrix.height,
            fixed, transpose);
}

template<typename T>
void DebugOutput::printColumnMajorMatrix(Matrix<T>& matrix, uint16_t width,
                                         uint16_t height, bool fixed, bool transpose)
{
    assert(width * height <= matrix.elements());

    if (fixed) std::cout << std::fixed;

    for (uint32_t i = 0; (transpose && i < width) || (!transpose && i < height); i++)
    {
        for (uint32_t j = 0; (transpose && j < height) || (!transpose && j < width); j++)
        {
            if (transpose) {
                std::cout << "  " << (matrix.data[i*height + j] >= 0 ? ' ' : '-')
                        << std::abs(matrix.data[i*height + j]);
            } else {
                std::cout << "  " << (matrix.data[j * height + i] >= 0 ? ' ' : '-')
                        << std::abs(matrix.data[j * height + i]);
            }
        }
        std::cout << std::endl;
    }

    if (fixed) std::cout.unsetf(std::ios_base::floatfield);
}
