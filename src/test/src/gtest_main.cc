#include <iostream>
#include <gtest/gtest.h>

#include <random>

static std::random_device RANDOM_DEVICE;
static std::mt19937 MT19937_GENERATOR(RANDOM_DEVICE());
static std::uniform_real_distribution<> UNIFORM_REAL(0.0f, 1.0f);

/* * /
#include "kernels/sortSelectColumns.h"
#include "kernels/sortSelectHeightAndRows.h"
#include "kernels/sortSelectRows.h"
#include "kernels/sortSelectWidthAndColumns.h"
#include "kernels/sumChanges.h"
#include "kernels/maxScore.h"
#include "utilities.h"
#include "secondStage.h"
#include "firstStage.h"
/* */
#include "wholeAlgo.h"

GTEST_API_ int main(int argc, char **argv)
{
    std::cout << "Running main() from gtest_main.cc" << std::endl;
    testing::InitGoogleTest(&argc, argv);
    auto res = RUN_ALL_TESTS();

    cudaDeviceReset();

    return res;
}
