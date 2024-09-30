#include <thrust/execution_policy.h>
#include <thrust/uninitialized_fill.h>

#include <unittest/unittest.h>

#ifdef THRUST_TEST_DEVICE_SIDE
template <typename ExecutionPolicy, typename Iterator, typename T>
__global__ void uninitialized_fill_kernel(ExecutionPolicy exec, Iterator first, Iterator last, T val)
{
  thrust::uninitialized_fill(exec, first, last, val);
}

template <typename ExecutionPolicy>
void TestUninitializedFillDevice(ExecutionPolicy exec)
{
  using Vector = thrust::device_vector<int>;
  using T      = Vector::value_type;

  Vector v{0, 1, 2, 3, 4};
  T sub(7);

  uninitialized_fill_kernel<<<1, 1>>>(exec, v.begin() + 1, v.begin() + 4, sub);
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  Vector ref{0, sub, sub, sub, 4};
  ASSERT_EQUAL(v, ref);

  sub = 8;

  uninitialized_fill_kernel<<<1, 1>>>(exec, v.begin() + 0, v.begin() + 3, sub);
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  ref = {sub, sub, sub, 7, 4};
  ASSERT_EQUAL(v, ref);

  sub = 9;

  uninitialized_fill_kernel<<<1, 1>>>(exec, v.begin() + 2, v.end(), sub);
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  ref = {8, 8, sub, sub, 9};
  ASSERT_EQUAL(v, ref);

  sub = 1;

  uninitialized_fill_kernel<<<1, 1>>>(exec, v.begin(), v.end(), sub);
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  ref = Vector(5, sub);
  ASSERT_EQUAL(v, ref);
}

void TestUninitializedFillDeviceSeq()
{
  TestUninitializedFillDevice(thrust::seq);
}
DECLARE_UNITTEST(TestUninitializedFillDeviceSeq);

void TestUninitializedFillDeviceDevice()
{
  TestUninitializedFillDevice(thrust::device);
}
DECLARE_UNITTEST(TestUninitializedFillDeviceDevice);
#endif

void TestUninitializedFillCudaStreams()
{
  using Vector = thrust::device_vector<int>;
  using T      = Vector::value_type;

  Vector v{0, 1, 2, 3, 4};
  T sub(7);

  cudaStream_t s;
  cudaStreamCreate(&s);

  thrust::uninitialized_fill(thrust::cuda::par.on(s), v.begin(), v.end(), sub);
  cudaStreamSynchronize(s);

  Vector ref(v.size(), sub);
  ASSERT_EQUAL(v, ref);

  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestUninitializedFillCudaStreams);

#ifdef THRUST_TEST_DEVICE_SIDE
template <typename ExecutionPolicy, typename Iterator1, typename Size, typename T, typename Iterator2>
__global__ void uninitialized_fill_n_kernel(ExecutionPolicy exec, Iterator1 first, Size n, T val, Iterator2 result)
{
  *result = thrust::uninitialized_fill_n(exec, first, n, val);
}

template <typename ExecutionPolicy>
void TestUninitializedFillNDevice(ExecutionPolicy exec)
{
  using Vector = thrust::device_vector<int>;
  using T      = Vector::value_type;

  Vector v{0, 1, 2, 3, 4};
  T sub(7);

  thrust::device_vector<Vector::iterator> iter_vec(1);

  uninitialized_fill_n_kernel<<<1, 1>>>(exec, v.begin() + 1, 3, sub, iter_vec.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  Vector::iterator iter = iter_vec[0];

  Vector ref{0, sub, sub, sub, 4};
  ASSERT_EQUAL(v, ref);
  ASSERT_EQUAL_QUIET(v.begin() + 4, iter);

  sub = 8;

  uninitialized_fill_n_kernel<<<1, 1>>>(exec, v.begin() + 0, 3, sub, iter_vec.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  iter = iter_vec[0];

  ref = {sub, sub, sub, 7, 4};
  ASSERT_EQUAL(v, ref);
  ASSERT_EQUAL_QUIET(v.begin() + 3, iter);

  sub = 9;

  uninitialized_fill_n_kernel<<<1, 1>>>(exec, v.begin() + 2, 3, sub, iter_vec.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  iter = iter_vec[0];

  ref = {8, 8, sub, sub, 9};
  ASSERT_EQUAL(v, ref);
  ASSERT_EQUAL_QUIET(v.end(), iter);

  sub = 1;

  uninitialized_fill_n_kernel<<<1, 1>>>(exec, v.begin(), v.size(), sub, iter_vec.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  iter = iter_vec[0];

  ref = Vector(5, sub);

  ASSERT_EQUAL(v, ref);
  ASSERT_EQUAL_QUIET(v.end(), iter);
}

void TestUninitializedFillNDeviceSeq()
{
  TestUninitializedFillNDevice(thrust::seq);
}
DECLARE_UNITTEST(TestUninitializedFillNDeviceSeq);

void TestUninitializedFillNDeviceDevice()
{
  TestUninitializedFillNDevice(thrust::device);
}
DECLARE_UNITTEST(TestUninitializedFillNDeviceDevice);
#endif

void TestUninitializedFillNCudaStreams()
{
  using Vector = thrust::device_vector<int>;
  using T      = Vector::value_type;

  Vector v{0, 1, 2, 3, 4};
  T sub(7);

  cudaStream_t s;
  cudaStreamCreate(&s);

  thrust::uninitialized_fill_n(thrust::cuda::par.on(s), v.begin(), v.size(), sub);
  cudaStreamSynchronize(s);

  Vector ref(5, sub);
  ASSERT_EQUAL(v, ref);

  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestUninitializedFillNCudaStreams);
