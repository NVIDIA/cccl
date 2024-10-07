#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/tabulate.h>

#include <unittest/unittest.h>

#ifdef THRUST_TEST_DEVICE_SIDE
template <typename ExecutionPolicy, typename Iterator, typename Function>
__global__ void tabulate_kernel(ExecutionPolicy exec, Iterator first, Iterator last, Function f)
{
  thrust::tabulate(exec, first, last, f);
}

template <typename ExecutionPolicy>
void TestTabulateDevice(ExecutionPolicy exec)
{
  using Vector = thrust::device_vector<int>;
  using namespace thrust::placeholders;
  using T = typename Vector::value_type;

  Vector v(5);

  tabulate_kernel<<<1, 1>>>(exec, v.begin(), v.end(), thrust::identity<T>());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  Vector ref{0, 1, 2, 3, 4};
  ASSERT_EQUAL(v, ref);

  tabulate_kernel<<<1, 1>>>(exec, v.begin(), v.end(), -_1);
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  ref = {0, -1, -2, -3, -4};
  ASSERT_EQUAL(v, ref);

  tabulate_kernel<<<1, 1>>>(exec, v.begin(), v.end(), _1 * _1 * _1);
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  ref = {0, 1, 8, 27, 64};
  ASSERT_EQUAL(v, ref);
}

void TestTabulateDeviceSeq()
{
  TestTabulateDevice(thrust::seq);
}
DECLARE_UNITTEST(TestTabulateDeviceSeq);

void TestTabulateDeviceDevice()
{
  TestTabulateDevice(thrust::device);
}
DECLARE_UNITTEST(TestTabulateDeviceDevice);
#endif

void TestTabulateCudaStreams()
{
  using namespace thrust::placeholders;
  using Vector = thrust::device_vector<int>;
  using T      = Vector::value_type;

  Vector v(5);

  cudaStream_t s;
  cudaStreamCreate(&s);

  thrust::tabulate(thrust::cuda::par.on(s), v.begin(), v.end(), thrust::identity<T>());
  cudaStreamSynchronize(s);

  Vector ref{0, 1, 2, 3, 4};
  ASSERT_EQUAL(v, ref);

  thrust::tabulate(thrust::cuda::par.on(s), v.begin(), v.end(), -_1);
  cudaStreamSynchronize(s);

  ref = {0, -1, -2, -3, -4};
  ASSERT_EQUAL(v, ref);

  thrust::tabulate(thrust::cuda::par.on(s), v.begin(), v.end(), _1 * _1 * _1);
  cudaStreamSynchronize(s);

  ref = {0, 1, 8, 27, 64};
  ASSERT_EQUAL(v, ref);

  cudaStreamSynchronize(s);
}
DECLARE_UNITTEST(TestTabulateCudaStreams);
