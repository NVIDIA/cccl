#include <thrust/execution_policy.h>
#include <thrust/swap.h>

#include <unittest/unittest.h>

#ifdef THRUST_TEST_DEVICE_SIDE
template <typename ExecutionPolicy, typename Iterator1, typename Iterator2>
__global__ void swap_ranges_kernel(ExecutionPolicy exec, Iterator1 first1, Iterator1 last1, Iterator2 first2)
{
  thrust::swap_ranges(exec, first1, last1, first2);
}

template <typename ExecutionPolicy>
void TestSwapRangesDevice(ExecutionPolicy exec)
{
  using Vector = thrust::device_vector<int>;

  Vector v1{0, 1, 2, 3, 4};
  Vector v2{5, 6, 7, 8, 9};
  Vector v1_ref(v2);
  Vector v2_ref(v1);

  swap_ranges_kernel<<<1, 1>>>(exec, v1.begin(), v1.end(), v2.begin());
  cudaError_t const err = cudaDeviceSynchronize();
  ASSERT_EQUAL(cudaSuccess, err);

  ASSERT_EQUAL(v1, v1_ref);
  ASSERT_EQUAL(v2, v2_ref);
}

void TestSwapRangesDeviceSeq()
{
  TestSwapRangesDevice(thrust::seq);
}
DECLARE_UNITTEST(TestSwapRangesDeviceSeq);

void TestSwapRangesDeviceDevice()
{
  TestSwapRangesDevice(thrust::device);
}
DECLARE_UNITTEST(TestSwapRangesDeviceDevice);
#endif

void TestSwapRangesCudaStreams()
{
  using Vector = thrust::device_vector<int>;

  Vector v1{0, 1, 2, 3, 4};
  Vector v2{5, 6, 7, 8, 9};
  Vector v1_ref(v2);
  Vector v2_ref(v1);

  cudaStream_t s;
  cudaStreamCreate(&s);

  thrust::swap_ranges(thrust::cuda::par.on(s), v1.begin(), v1.end(), v2.begin());
  cudaStreamSynchronize(s);

  ASSERT_EQUAL(v1, v1_ref);
  ASSERT_EQUAL(v2, v2_ref);

  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestSwapRangesCudaStreams);
