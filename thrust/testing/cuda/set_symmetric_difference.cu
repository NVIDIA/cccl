#include <thrust/execution_policy.h>
#include <thrust/set_operations.h>

#include <unittest/unittest.h>

#ifdef THRUST_TEST_DEVICE_SIDE
template <typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Iterator3, typename Iterator4>
__global__ void set_symmetric_difference_kernel(
  ExecutionPolicy exec,
  Iterator1 first1,
  Iterator1 last1,
  Iterator2 first2,
  Iterator2 last2,
  Iterator3 result1,
  Iterator4 result2)
{
  *result2 = thrust::set_symmetric_difference(exec, first1, last1, first2, last2, result1);
}

template <typename ExecutionPolicy>
void TestSetSymmetricDifferenceDevice(ExecutionPolicy exec)
{
  using Vector   = thrust::device_vector<int>;
  using Iterator = typename Vector::iterator;

  Vector a{0, 2, 4, 6}, b{0, 3, 3, 4, 7};

  Vector ref{2, 3, 3, 6, 7};
  Vector result(5);
  thrust::device_vector<Iterator> end_vec(1);

  set_symmetric_difference_kernel<<<1, 1>>>(
    exec, a.begin(), a.end(), b.begin(), b.end(), result.begin(), end_vec.begin());
  cudaError_t const err = cudaDeviceSynchronize();
  ASSERT_EQUAL(cudaSuccess, err);

  Iterator end = end_vec[0];

  ASSERT_EQUAL_QUIET(result.end(), end);
  ASSERT_EQUAL(ref, result);
}

void TestSetSymmetricDifferenceDeviceSeq()
{
  TestSetSymmetricDifferenceDevice(thrust::seq);
}
DECLARE_UNITTEST(TestSetSymmetricDifferenceDeviceSeq);

void TestSetSymmetricDifferenceDeviceDevice()
{
  TestSetSymmetricDifferenceDevice(thrust::device);
}
DECLARE_UNITTEST(TestSetSymmetricDifferenceDeviceDevice);
#endif

void TestSetSymmetricDifferenceCudaStreams()
{
  using Vector   = thrust::device_vector<int>;
  using Iterator = Vector::iterator;

  Vector a{0, 2, 4, 6}, b{0, 3, 3, 4, 7};

  Vector ref{2, 3, 3, 6, 7};
  Vector result(5);

  cudaStream_t s;
  cudaStreamCreate(&s);

  Iterator end =
    thrust::set_symmetric_difference(thrust::cuda::par.on(s), a.begin(), a.end(), b.begin(), b.end(), result.begin());
  cudaStreamSynchronize(s);

  ASSERT_EQUAL_QUIET(result.end(), end);
  ASSERT_EQUAL(ref, result);

  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestSetSymmetricDifferenceCudaStreams);
