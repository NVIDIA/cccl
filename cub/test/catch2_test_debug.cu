#include <cub/util_debug.cuh>
#include <cub/util_device.cuh>

#include "cub_test_macros.h"

CUB_TEST_CASE("CubDebug returns input error", "[debug][utils]", CUB_SMALL)
{
  REQUIRE(CubDebug(cudaSuccess) == cudaSuccess);
  REQUIRE(CubDebug(cudaErrorInvalidConfiguration) == cudaErrorInvalidConfiguration);
}

CUB_TEST_CASE("CubDebug returns new errors", "[debug][utils]", CUB_SMALL)
{
  cub::detail::EmptyKernel<int><<<0, 0>>>();
  cudaError error = cudaPeekAtLastError();

  REQUIRE(error != cudaSuccess);
  REQUIRE(CubDebug(cudaSuccess) != cudaSuccess);
}

CUB_TEST_CASE("CubDebug prefers input errors", "[debug][utils]", CUB_SMALL)
{
  cub::detail::EmptyKernel<int><<<0, 0>>>();
  cudaError error = cudaPeekAtLastError();

  REQUIRE(error != cudaSuccess);
  REQUIRE(CubDebug(cudaErrorMemoryAllocation) != cudaSuccess);
}

CUB_TEST_CASE("CubDebug resets last error", "[debug][utils]", CUB_SMALL)
{
  cub::detail::EmptyKernel<int><<<0, 0>>>();
  cudaError error = cudaPeekAtLastError();

  REQUIRE(error != cudaSuccess);
  REQUIRE(CubDebug(cudaSuccess) != cudaSuccess);
  REQUIRE(CubDebug(cudaSuccess) == cudaSuccess);
}
