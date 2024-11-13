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
  ASSERT_EQUAL(cudaSuccess, err);

  Vector ref{0, 1, 2, 3, 4};
  ASSERT_EQUAL(v2, ref);
}

void TestUninitializedCopyDeviceSeq()
{
  TestUninitializedCopyDevice(thrust::seq);
}
DECLARE_UNITTEST(TestUninitializedCopyDeviceSeq);

void TestUninitializedCopyDeviceDevice()
{
  TestUninitializedCopyDevice(thrust::device);
}
DECLARE_UNITTEST(TestUninitializedCopyDeviceDevice);
#endif

void TestUninitializedCopyCudaStreams()
{
  using Vector = thrust::device_vector<int>;

  Vector v1{0, 1, 2, 3, 4};

  // copy to Vector
  Vector v2(5);

  cudaStream_t s;
  cudaStreamCreate(&s);

  thrust::uninitialized_copy(thrust::cuda::par.on(s), v1.begin(), v1.end(), v2.begin());
  cudaStreamSynchronize(s);

  ASSERT_EQUAL(v2, v1);
  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestUninitializedCopyCudaStreams);

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
  ASSERT_EQUAL(cudaSuccess, err);

  ASSERT_EQUAL(v2, v1);
}

void TestUninitializedCopyNDeviceSeq()
{
  TestUninitializedCopyNDevice(thrust::seq);
}
DECLARE_UNITTEST(TestUninitializedCopyNDeviceSeq);

void TestUninitializedCopyNDeviceDevice()
{
  TestUninitializedCopyNDevice(thrust::device);
}
DECLARE_UNITTEST(TestUninitializedCopyNDeviceDevice);
#endif

void TestUninitializedCopyNCudaStreams()
{
  using Vector = thrust::device_vector<int>;

  Vector v1{0, 1, 2, 3, 4};

  // copy to Vector
  Vector v2(5);

  cudaStream_t s;
  cudaStreamCreate(&s);

  thrust::uninitialized_copy_n(thrust::cuda::par.on(s), v1.begin(), v1.size(), v2.begin());
  cudaStreamSynchronize(s);
  ASSERT_EQUAL(v2, v1);

  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestUninitializedCopyNCudaStreams);
