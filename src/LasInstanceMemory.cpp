#include "LasInstanceMemory.h"

#include <algorithm>
#include "ErrorHandling.h"

uint32_t LasInstanceMemory::nextPowerOfTwo(uint32_t i)
{
    uint32_t nextPowerOfTwo;
    for (nextPowerOfTwo = 1; nextPowerOfTwo < i; nextPowerOfTwo <<= 1)
    {
    };

    return nextPowerOfTwo;
}

LasInstanceMemory::LasInstanceMemory(const double* matrix,
                                     const uint32_t& width,
                                     const uint32_t& height,
                                     const uint32_t& InvocationsPerBicluster)
        : Width(width),
          NextPowerOfTwoAfterWidth(nextPowerOfTwo(width)),
          Height(height),
          NextPowerOfTwoAfterHeight(nextPowerOfTwo(height)),
          MaxDimension(std::max(Width, Height)),
          InvocationsPerBicluster(InvocationsPerBicluster),

          matrix(this->Width, this->Height, MatrixLocation::Host),
          deviceMatrix(this->Width, this->Height, MatrixLocation::Device),

          columnSet(InvocationsPerBicluster, this->Width, MatrixLocation::Host),
          deviceColumnSet(InvocationsPerBicluster, this->Width,
                  MatrixLocation::Device),

          rowSet(this->Height, InvocationsPerBicluster, MatrixLocation::Host),
          deviceRowSet(this->Height, InvocationsPerBicluster,
                  MatrixLocation::Device),

          sums(InvocationsPerBicluster * MaxDimension, 1, MatrixLocation::Host),
          deviceSums(InvocationsPerBicluster * MaxDimension, 1,
                  MatrixLocation::Device),

          deviceBiclusterScores(InvocationsPerBicluster, 1,
                  MatrixLocation::Device),

          sizes(InvocationsPerBicluster, 2, MatrixLocation::Host),
          deviceSizes(InvocationsPerBicluster, 2, MatrixLocation::Device),

          deviceLgamma(MaxDimension + 2, 1, MatrixLocation::Device),

          deviceColumnChanges(InvocationsPerBicluster, 1, MatrixLocation::Device),
          deviceRowChanges(InvocationsPerBicluster, 1, MatrixLocation::Device),
          deviceInvocationsPermutation(InvocationsPerBicluster, 1, MatrixLocation::Device)
{
    // copying matrix
    this->matrix.allocate();
    std::fill(this->matrix.begin(), this->matrix.end(), 0.0);
    for (uint32_t i = 0; i < width; ++i)
    {
        std::copy(matrix + i * height, matrix + (i + 1) * height,
                this->matrix.begin() + i * this->Height);
    }

}

// Separating this to correctly handle all bad_allocs
void LasInstanceMemory::init()
{

    deviceMatrix.allocate();
    matrix.copyTo(deviceMatrix);

    columnSet.allocate();
    deviceColumnSet.allocate();
    deviceMemory.ColumnSet = deviceColumnSet.data;

    rowSet.allocate();
    deviceRowSet.allocate();
    deviceMemory.RowSet = deviceRowSet.data;

    sums.allocate();
    deviceSums.allocate();
    deviceMemory.Sums = deviceSums.data;

    deviceBiclusterScores.allocate();
    deviceMemory.Scores = deviceBiclusterScores.data;

    sizes.allocate();
    deviceSizes.allocate();
    deviceMemory.Sizes = deviceSizes.data;

    deviceLgamma.allocate();
    deviceMemory.gammaln = deviceLgamma.data;

    deviceColumnChanges.allocate();
    deviceMemory.ColumnChanges = deviceColumnChanges.data;
    deviceRowChanges.allocate();
    deviceMemory.RowChanges = deviceRowChanges.data;

    deviceInvocationsPermutation.allocate();

    checkCudaErrors(cudaMemset(deviceRowSet.data, 0, deviceRowSet.size()));

    double* gammaln = new double[MaxDimension + 2];
    for (uint32_t i = 1; i < MaxDimension + 2; i++)
    {
        gammaln[i] = std::lgamma(double(i));
    }
    checkCudaErrors(
            cudaMemcpy(deviceLgamma.data, gammaln, deviceLgamma.size(),
                    cudaMemcpyHostToDevice));
    delete[] gammaln;
}

LasInstanceMemory::~LasInstanceMemory()
{
}
