#include <thrust/system/cuda/detail/util.h>

#include <unittest/unittest.h>

// These tests verify that trivial_copy_from_device and trivial_copy_to_device
// use cudaMemcpyDefault instead of cudaMemcpyDeviceToHost/cudaMemcpyHostToDevice.
//
// With explicit copy directions, cudaMemcpyAsync fails when the actual memory
// location doesn't match the expected direction (e.g., cudaMemcpyDeviceToHost
// with a host source pointer returns cudaErrorInvalidValue).
//
// With cudaMemcpyDefault, CUDA determines the correct direction at runtime,
// which is necessary when device-accessible pointers may point to host memory.

void TestTrivialCopyFromDevice_HostSource()
{
  int src[]  = {0, 10, 20, 30, 40};
  int dst[5] = {};

  const cudaError_t status = thrust::cuda_cub::trivial_copy_from_device(dst, src, 5, cudaStreamDefault);

  REQUIRE(status == cudaSuccess);
  for (int i = 0; i < 5; i++)
  {
    REQUIRE(dst[i] == i * 10);
  }
}
TEST_CASE("TestTrivialCopyFromDevice_HostSource", "[trivial_copy_memcpy_default]")
{
  TestTrivialCopyFromDevice_HostSource();
}

void TestTrivialCopyToDevice_HostDest()
{
  int src[]  = {0, 100, 200, 300, 400};
  int dst[5] = {};

  const cudaError_t status = thrust::cuda_cub::trivial_copy_to_device(dst, src, 5, cudaStreamDefault);

  REQUIRE(status == cudaSuccess);
  for (int i = 0; i < 5; i++)
  {
    REQUIRE(dst[i] == i * 100);
  }
}
TEST_CASE("TestTrivialCopyToDevice_HostDest", "[trivial_copy_memcpy_default]")
{
  TestTrivialCopyToDevice_HostDest();
}
