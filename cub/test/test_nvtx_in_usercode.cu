#include <cub/device/device_for.cuh> // internal include of NVTX

#include <thrust/iterator/counting_iterator.h>

#include <cuda/iterator>
#include <cuda/std/functional>

#include "cub_non_catch2_test_memory.h"
#include <nvtx3/nvtx3.hpp> // user-side include of NVTX, retrieved elsewhere

CUB_TEST_MEMORY_CLASS(CUB_SMALL);

int main()
{
  nvtx3::scoped_range range("user-range"); // user-side use of unversioned NVTX API

  cuda::counting_iterator<int> it{0};
  cub::DeviceFor::ForEach(it, it + 16, ::cuda::std::negate<int>{}); // internal use of NVTX
  cudaDeviceSynchronize();
}
