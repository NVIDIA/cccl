#include <thrust/execution_policy.h>
#include <thrust/set_operations.h>

#include <unittest/unittest.h>

#ifdef THRUST_TEST_DEVICE_SIDE
template <typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Iterator3, typename Iterator4>
__global__ void set_difference_kernel(
  ExecutionPolicy exec,
  Iterator1 first1,
  Iterator1 last1,
  Iterator2 first2,
  Iterator2 last2,
  Iterator3 result1,
  Iterator4 result2)
{
  *result2 = thrust::set_difference(exec, first1, last1, first2, last2, result1);
}

template <typename ExecutionPolicy>
void TestSetDifferenceDevice(ExecutionPolicy exec)
{
  using Vector   = thrust::device_vector<int>;
  using Iterator = typename Vector::iterator;

  Vector a{0, 2, 4, 5}, b{0, 3, 3, 4, 6};

  Vector ref{2, 5};
  Vector result(2);

  thrust::device_vector<Iterator> end_vec(1);

  set_difference_kernel<<<1, 1>>>(exec, a.begin(), a.end(), b.begin(), b.end(), result.begin(), end_vec.begin());
  cudaError_t const err = cudaDeviceSynchronize();
  ASSERT_EQUAL(cudaSuccess, err);

  Iterator end = end_vec.front();

  ASSERT_EQUAL_QUIET(result.end(), end);
  ASSERT_EQUAL(ref, result);
}

void TestSetDifferenceDeviceSeq()
{
  TestSetDifferenceDevice(thrust::seq);
}
DECLARE_UNITTEST(TestSetDifferenceDeviceSeq);

void TestSetDifferenceDeviceDevice()
{
  TestSetDifferenceDevice(thrust::device);
}
DECLARE_UNITTEST(TestSetDifferenceDeviceDevice);
#endif

void TestSetDifferenceCudaStreams()
{
  using Vector   = thrust::device_vector<int>;
  using Iterator = Vector::iterator;

  Vector a{0, 2, 4, 5}, b{0, 3, 3, 4, 6};

  Vector ref{2, 5};
  Vector result(2);

  cudaStream_t s;
  cudaStreamCreate(&s);

  Iterator end =
    thrust::set_difference(thrust::cuda::par.on(s), a.begin(), a.end(), b.begin(), b.end(), result.begin());
  cudaStreamSynchronize(s);

  ASSERT_EQUAL_QUIET(result.end(), end);
  ASSERT_EQUAL(ref, result);

  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestSetDifferenceCudaStreams);
