#pragma once

#include <stdint.h>
#include <vector>

// Result of LAS algorithm
struct Bicluster
{
    uint16_t width = 0;
    uint16_t height = 0;
    double score = 0.0;
    std::vector<uint8_t> selectedRows;
    std::vector<uint8_t> selectedColumns;
};
