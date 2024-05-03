#include <cub/device/device_for.cuh> // internal include of NVTX

#include <thrust/iterator/counting_iterator.h>

#include <nvtx3/nvtx3.hpp> // user-side include of NVTX, retrieved elsewhere

struct Op
{
  _CCCL_HOST_DEVICE void operator()(int i) const
  {
    printf("%d\n", i);
  }
};

int main()
{
  nvtx3::scoped_range range("user-range"); // user-side use of NVTX

  thrust::counting_iterator<int> it{0};
  cub::DeviceFor::ForEach(it, it + 16, Op{}); // internal use of NVTX
  cudaDeviceSynchronize();
}
