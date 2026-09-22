#include <thrust/execution_policy.h>
#include <thrust/uninitialized_copy.h>

#include <unittest/unittest.h>

#ifdef THRUST_TEST_DEVICE_SIDE
template <typename ExecutionPolicy, typename Iterator1, typename Iterator2>
__global__ void uninitialized_copy_kernel(ExecutionPolicy exec, Iterator1 first, Iterator1 last, Iterator2 result)
{
  thrust::uninitialized_copy(exec, first, last, result);
}

template <typename ExecutionPolicy>
void TestUninitializedCopyDevice(ExecutionPolicy exec)
{
  using Vector = thrust::device_vector<int>;

  Vector v1{0, 1, 2, 3, 4};

  // copy to Vector
  Vector v2(5);
  uninitialized_copy_kernel<<<1, 1>>>(exec, v1.begin(), v1.end(), v2.begin());
  cudaError_t const err = cudaDeviceSynchronize();
  REQUIRE(cudaSuccess == err);

  Vector ref{0, 1, 2, 3, 4};
  REQUIRE(v2 == ref);
}

TEST_CASE("TestUninitializedCopyDeviceSeq", "[uninitialized_copy]")
{
  TestUninitializedCopyDevice(thrust::seq);
}

TEST_CASE("TestUninitializedCopyDeviceDevice", "[uninitialized_copy]")
{
  TestUninitializedCopyDevice(thrust::device);
}
#endif

TEST_CASE("TestUninitializedCopyCudaStreams", "[uninitialized_copy]")
{
  using Vector = thrust::device_vector<int>;

  Vector v1{0, 1, 2, 3, 4};

  // copy to Vector
  Vector v2(5);

  cudaStream_t s;
  cudaStreamCreate(&s);

  thrust::uninitialized_copy(thrust::cuda::par.on(s), v1.begin(), v1.end(), v2.begin());
  cudaStreamSynchronize(s);

  REQUIRE(v2 == v1);
  cudaStreamDestroy(s);
}

#ifdef THRUST_TEST_DEVICE_SIDE
template <typename ExecutionPolicy, typename Iterator1, typename Size, typename Iterator2>
__global__ void uninitialized_copy_n_kernel(ExecutionPolicy exec, Iterator1 first, Size n, Iterator2 result)
{
  thrust::uninitialized_copy_n(exec, first, n, result);
}

template <typename ExecutionPolicy>
void TestUninitializedCopyNDevice(ExecutionPolicy exec)
{
  using Vector = thrust::device_vector<int>;

  Vector v1{0, 1, 2, 3, 4};

  // copy to Vector
  Vector v2(5);
  uninitialized_copy_n_kernel<<<1, 1>>>(exec, v1.begin(), v1.size(), v2.begin());
  cudaError_t const err = cudaDeviceSynchronize();
  REQUIRE(cudaSuccess == err);

  REQUIRE(v2 == v1);
}

TEST_CASE("TestUninitializedCopyNDeviceSeq", "[uninitialized_copy]")
{
  TestUninitializedCopyNDevice(thrust::seq);
}

TEST_CASE("TestUninitializedCopyNDeviceDevice", "[uninitialized_copy]")
{
  TestUninitializedCopyNDevice(thrust::device);
}
#endif

TEST_CASE("TestUninitializedCopyNCudaStreams", "[uninitialized_copy]")
{
  using Vector = thrust::device_vector<int>;

  Vector v1{0, 1, 2, 3, 4};

  // copy to Vector
  Vector v2(5);

  cudaStream_t s;
  cudaStreamCreate(&s);

  thrust::uninitialized_copy_n(thrust::cuda::par.on(s), v1.begin(), v1.size(), v2.begin());
  cudaStreamSynchronize(s);
  REQUIRE(v2 == v1);

  cudaStreamDestroy(s);
}
