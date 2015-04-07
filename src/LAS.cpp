#include "LAS.h"

#include <chrono>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include "ErrorHandling.h"
#include "LasException.h"
#include "kernels/kernels.h"

static const double alpha = 1.0;
static const double beta = 0.0;

LAS::LAS(int cudaDeviceId, cublasHandle_t cublasHandle, uint16_t maxBiclusters,
         uint16_t InvocationsPerBicluster, double scoreThreshold)
        : cudaDeviceId(cudaDeviceId), cublasHandle(cublasHandle),
          cublasCreated(false), maxBiclusters(maxBiclusters),
          invocationsPerBicluster(InvocationsPerBicluster),
          scoreThreshold(scoreThreshold), MT19937_GENERATOR(RANDOM_DEVICE()),
          UNIFORM_REAL(0.0f, 1.0f)
{
    if (cublasHandle == nullptr)
    {
        checkCudaErrors(cublasCreate(&this->cublasHandle));
        cublasCreated = true;
    }
}

LAS::~LAS()
{
    if (cublasCreated)
    {
        cublasDestroy(cublasHandle);
    }
}

uint32_t LAS::getInvocationsPerBicluster() const
{
    return invocationsPerBicluster;
}

void LAS::setInvocationsPerBicluster(uint32_t invocationsPerBicluster)
{
    this->invocationsPerBicluster = invocationsPerBicluster;
}

uint16_t LAS::getMaxBiclusters() const
{
    return maxBiclusters;
}

void LAS::setMaxBiclusters(uint16_t maxBiclusters)
{
    this->maxBiclusters = maxBiclusters;
}

double LAS::getScoreThreshold() const
{
    return scoreThreshold;
}

void LAS::setScoreThreshold(double scoreThreshold)
{
    this->scoreThreshold = scoreThreshold;
}

std::vector<Bicluster> LAS::run(const double* matrix, uint16_t width,
                                uint16_t height) throw (LasException,
                                                std::bad_alloc)
{
    std::vector<Bicluster> result;

    checkArguments(matrix, width, height);

    LasInstanceMemory memory(matrix, width, height, invocationsPerBicluster);
    memory.init();

    // used for initialization of column set at each iteration
    std::vector<uint16_t> columnsPermutations(memory.columnSet.elements());

    for (int k = 0; k < maxBiclusters; ++k)
    {
        initializeIteration(columnsPermutations, memory);


        firstStageSearch(memory);

        secondStageSearch(memory);


        // after second stage scores are written to first InvocationsPerBicluster elements of device sums matrix
        std::pair<uint32_t, double> maximum = maxScore(
                memory.deviceBiclusterScores);
        std::pair<uint16_t, uint16_t> size = extractBicluster(memory, maximum,
                result);

        if (maximum.second < scoreThreshold)
        {
            break;
        }

        if (k < maxBiclusters - 1)
        {
            subtractAverage(memory, maximum.first, size);
        }
    }

    return result;
}

// checks ability to execute algorithm
void LAS::checkArguments(const double* matrix, uint32_t width, uint32_t height)
{
    if (matrix == nullptr)
    {
        throw LasException("Null input matrix");
    }

    if (width % 2 != 0 or height % 2 != 0)
    {
        throw LasException("Dimensions must be even.");
    }

    if (invocationsPerBicluster < 1 or maxBiclusters < 1 or width * height < 1)
    {
        throw LasException("Empty task");
    }

    if (invocationsPerBicluster > width * height)
    {
        throw LasException("Wellll, it's complicated.");
    }

    uint32_t maxDimension = std::max(width, height);
    std::size_t requiredGlobalMemory = sizeof(double) * width * height // matrix
    + sizeof(double) * width * invocationsPerBicluster // columns set
    + sizeof(double) * height * invocationsPerBicluster // rows set
    + sizeof(double) * maxDimension * invocationsPerBicluster // partial sums
    + sizeof(double) * invocationsPerBicluster // biclusters scores
    + sizeof(uint16_t) * 2 * invocationsPerBicluster // sizes
    + sizeof(double) * (maxDimension + 2) // lgamma or gammaln
    + sizeof(uint32_t) * invocationsPerBicluster // columns changes
    + sizeof(uint32_t) * invocationsPerBicluster // rows changes
            ;

    std::size_t freeMemory, totalMemory;
    checkCudaErrors(cudaMemGetInfo(&freeMemory, &totalMemory));
    if (freeMemory < requiredGlobalMemory)
    {
        throw LasException("Not enough free device memory");
    }

    cudaDeviceProp deviceProperties;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProperties, cudaDeviceId));

    uint32_t nextPowerOfTwoAfterMaxDimension =
            LasInstanceMemory::nextPowerOfTwo(maxDimension);
    std::size_t requiredSharedMemory =
            sizeof(double) * nextPowerOfTwoAfterMaxDimension
            + sizeof(uint16_t) * nextPowerOfTwoAfterMaxDimension
            + sizeof(uint16_t) + sizeof(double) + sizeof(bool);
    if (deviceProperties.sharedMemPerBlock < requiredSharedMemory)
    {
        throw LasException("Not enough shared memory");
    }

    if (deviceProperties.maxThreadsPerBlock < maxDimension / 2)
    {
        throw LasException("Not enough threads per block");
    }
}

void LAS::initializeIteration(std::vector<uint16_t>& columnsPermutations,
                              LasInstanceMemory& memory)
{
    // generating initial bicluster sizes
    bool widthOrHeight = true;
    std::generate(memory.sizes.begin(), memory.sizes.end(),
            [&]
            {   uint32_t max = widthOrHeight? memory.Width : memory.Height;
                widthOrHeight = not widthOrHeight;
                return std::floor(std::pow(UNIFORM_REAL(MT19937_GENERATOR), 2)*double(max) / 2.0) + 1;
            });
    memory.sizes.copyTo(memory.deviceSizes);

    // initializing permutations
    for (int i = columnsPermutations.size() - 1; i >= 0; --i)
    {
        columnsPermutations[i] = i % memory.Width;
    }

    std::fill(memory.columnSet.begin(), memory.columnSet.end(), 0.0);
    for (uint32_t j = 0; j < invocationsPerBicluster; j++)
    {
        // shuffling permutation for each invocation
        std::shuffle(columnsPermutations.begin() + j * memory.Width,
                columnsPermutations.begin() + (j + 1) * memory.Width,
                MT19937_GENERATOR);

        // marking first k columns as selected,
        // where k is previously selected width of bicluster
        for (uint16_t i = 0; i < memory.sizes.begin()[j * 2]; i++)
        {
            memory.columnSet.begin()[j * memory.Width
                    + columnsPermutations[j * memory.Width + i]] = 1;
        }
    }
    memory.columnSet.copyTo(memory.deviceColumnSet);

    checkCudaErrors(
            cudaMemset(memory.deviceRowSet.data, 0,
                    memory.deviceRowSet.size()));
}

// extracts best bicluster of iteration and adds it to biclusters vector
// also returns bicluster size
std::pair<uint16_t, uint16_t> LAS::extractBicluster(
        LasInstanceMemory& memory,
        std::pair<uint32_t, double> biclusterIndexAndScore,
        std::vector<Bicluster>& biclusters)
{
    uint32_t bestBicluster = biclusterIndexAndScore.first;

    Bicluster bicluster;
    bicluster.score = biclusterIndexAndScore.second;

    uint16_t size[2];
    checkCudaErrors(
            cudaMemcpy(size, memory.deviceSizes.begin() + 2 * bestBicluster,
                    2 * sizeof(uint16_t), cudaMemcpyDeviceToHost));

    bicluster.width = size[0];
    bicluster.height = size[1];

    double* selectedArray = new double[memory.MaxDimension];

    checkCudaErrors(
            cudaMemcpy(selectedArray,
                    memory.deviceColumnSet.begin()
                            + bestBicluster * memory.matrix.width,
                    sizeof(double) * memory.MaxDimension,
                    cudaMemcpyDeviceToHost));
    bicluster.selectedColumns.resize(memory.matrix.width);
    for (auto i = 0; i < bicluster.selectedColumns.size(); ++i)
    {
        bicluster.selectedColumns[i] = uint8_t(selectedArray[i] > 0.0);
    }

    // some magic to copy selected rows in one cuda call
    checkCudaErrors(
            cudaMemcpy2D(selectedArray, sizeof(double),
                    memory.deviceRowSet.begin() + bestBicluster,
                    invocationsPerBicluster * sizeof(double), sizeof(double),
                    memory.matrix.height, cudaMemcpyDeviceToHost));
    bicluster.selectedRows.resize(memory.matrix.height);
    for (auto i = 0; i < bicluster.selectedRows.size(); ++i)
    {
        bicluster.selectedRows[i] = uint8_t(selectedArray[i] > 0.0);
    }

    delete[] selectedArray;

    biclusters.push_back(bicluster);

    return std::make_pair(size[0], size[1]);
}

// subtracts bicluster average from matrix in elements of this bicluster
void LAS::subtractAverage(LasInstanceMemory& memory, uint32_t bestBicluster,
                          const std::pair<uint16_t, uint16_t>& size)
{

    checkCudaErrors(
            cublasDgemv(cublasHandle, CUBLAS_OP_N,
                    memory.deviceMatrix.height, memory.deviceMatrix.width,
                    &alpha,
                    memory.deviceMatrix.begin(), memory.deviceMatrix.height,
                    memory.deviceColumnSet.begin() + bestBicluster*memory.deviceColumnSet.height,
                    1, &beta,
                    memory.deviceSums.begin(), 1));

    double average;
    checkCudaErrors(
            cublasDdot(cublasHandle,
                    memory.deviceMatrix.height, memory.deviceSums.begin(),
                    1, memory.deviceRowSet.begin() + bestBicluster,
                    invocationsPerBicluster, &average));
    average /= -double(size.first) * double(size.second);

    checkCudaErrors(
            cublasDger(cublasHandle, memory.deviceMatrix.height, memory.deviceMatrix.width,
                    &average,
                    memory.deviceRowSet.begin() + bestBicluster, invocationsPerBicluster,
                    memory.deviceColumnSet.begin() + bestBicluster*memory.deviceColumnSet.height,
                    1, memory.deviceMatrix.begin(), memory.deviceMatrix.height));
}

// search with constant bicluster size
void LAS::firstStageSearch(LasInstanceMemory& memory)
{
    uint32_t changes;

    checkCudaErrors(
            cudaMemset(memory.deviceColumnChanges.begin(), 1,
                    memory.deviceColumnChanges.size()));
    do
    {

        checkCudaErrors(
                cublasDgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                        memory.deviceMatrix.height, invocationsPerBicluster, memory.deviceMatrix.width,
                        &alpha,
                        memory.deviceMatrix.begin(), memory.deviceMatrix.height,
                        memory.deviceColumnSet.begin(), memory.deviceColumnSet.height,
                        &beta,
                        memory.deviceSums.begin(), memory.deviceMatrix.height ));

        checkCudaErrors(
                cudaMemset(memory.deviceRowChanges.begin(), 0,
                        memory.deviceRowChanges.size()));

        sortSelectRows(memory);

        checkCudaErrors(
                cublasDgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                        invocationsPerBicluster, memory.deviceMatrix.width, memory.deviceMatrix.height,
                        &alpha,
                        memory.deviceRowSet.begin(), invocationsPerBicluster,
                        memory.deviceMatrix.begin(), memory.deviceMatrix.height,
                        &beta,
                        memory.deviceSums.begin(), invocationsPerBicluster ));

        checkCudaErrors(
                cudaMemset(memory.deviceColumnChanges.begin(), 0,
                        memory.deviceColumnChanges.size()));

        sortSelectColumns(memory);

        changes = reduceChanges(memory.deviceColumnChanges);

    } while (changes > 0);
}

void LAS::secondStageSearch(LasInstanceMemory& memory)
{

    uint32_t changes = 0;

    checkCudaErrors(
            cudaMemset(memory.deviceColumnChanges.begin(), 1,
                    memory.deviceColumnChanges.size()));

    do
    {
        checkCudaErrors(
                cublasDgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                        memory.deviceMatrix.height, invocationsPerBicluster, memory.deviceMatrix.width,
                        &alpha,
                        memory.deviceMatrix.begin(), memory.deviceMatrix.height,
                        memory.deviceColumnSet.begin(), memory.deviceColumnSet.height,
                        &beta,
                        memory.deviceSums.begin(), memory.deviceMatrix.height ));

        checkCudaErrors(
                cudaMemset(memory.deviceRowChanges.begin(), 0,
                        memory.deviceRowChanges.size()));

        sortSelectHeightAndRows(memory);

        checkCudaErrors(
                cublasDgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                        invocationsPerBicluster, memory.deviceMatrix.width, memory.deviceMatrix.height,
                        &alpha,
                        memory.deviceRowSet.begin(), invocationsPerBicluster,
                        memory.deviceMatrix.begin(), memory.deviceMatrix.height,
                        &beta,
                        memory.deviceSums.begin(), invocationsPerBicluster ));

        checkCudaErrors(
                cudaMemset(memory.deviceColumnChanges.begin(), 0,
                        memory.deviceColumnChanges.size()));

        sortSelectWidthAndColumns(memory);

        changes = reduceChanges(memory.deviceColumnChanges);

    } while (changes > 0);

}
