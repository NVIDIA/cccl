// The purpose of this test is to verify that CUB can use NVTX without any additional dependencies. It is built as part
// of the unit tests, but can also be built standalone:

// Compile (from current directory):
//   nvcc test_nvtx_standalone.cu -I../../cub -I../../thrust -I../../libcudacxx/include -o nvtx_standalone
// Profile & view:
//   (nsys profile -o nvtx_standalone.nsys-rep -f true ./nvtx_standalone || true) && nsys-ui nvtx_standalone.nsys-rep

#include <cub/device/device_for.cuh>

#include <cuda/iterator>
#include <cuda/std/functional>

int main()
{
  _CCCL_NVTX_RANGE_SCOPE("main");

  cuda::counting_iterator<int> it{0};
  cub::DeviceFor::ForEach(it, it + 16, ::cuda::std::negate<int>{});
  cudaDeviceSynchronize();
}
