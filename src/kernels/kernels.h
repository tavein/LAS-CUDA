#pragma once

#include <utility>

#include "../LasInstanceMemory.h"

void sortSelectColumns(LasInstanceMemory& memory);

void sortSelectHeightAndRows(LasInstanceMemory& memory);

void sortSelectRows(LasInstanceMemory& memory);

void sortSelectWidthAndColumns(LasInstanceMemory& memory);

uint32_t reduceChanges(Matrix<uint32_t>& deviceChanges);

std::pair<uint32_t, double> maxScore(Matrix<double>& deviceSums);
