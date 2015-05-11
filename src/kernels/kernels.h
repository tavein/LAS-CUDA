#pragma once

#include <utility>

#include "../LasInstanceMemory.h"

void sortSelectColumns(LasInstanceMemory& memory, uint32_t activeInvocations);

void sortSelectHeightAndRows(LasInstanceMemory& memory, uint32_t activeInvocations);

void sortSelectRows(LasInstanceMemory& memory, uint32_t activeInvocations);

void sortSelectWidthAndColumns(LasInstanceMemory& memory, uint32_t activeInvocations);

std::pair<uint32_t, double> maxScore(Matrix<double>& deviceSums);


void reorderRowSets(LasInstanceMemory& memory, uint32_t activeInvocationsPerBicluster);

void reorderColumnSets(LasInstanceMemory& memory, uint32_t activeInvocationsPerBicluster);

void reorderSizesAndScores(LasInstanceMemory& memory, uint32_t activeInvocationsPerBicluster);

uint32_t sortSelectActiveInvocations(Matrix<uint32_t>& changes, Matrix<uint16_t>& permutation,
                                    uint32_t activeInvocations);
