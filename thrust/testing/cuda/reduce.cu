#include <thrust/execution_policy.h>
#include <thrust/reduce.h>

#include <cuda/iterator>

#include <unittest/unittest.h>

template <typename ExecutionPolicy, typename Iterator, typename T, typename Iterator2>
__global__ void reduce_kernel(ExecutionPolicy exec, Iterator first, Iterator last, T init, Iterator2 result)
{
  *result = thrust::reduce(exec, first, last, init);
}

#ifdef THRUST_TEST_DEVICE_SIDE
template <typename T, typename ExecutionPolicy>
void TestReduceDevice(ExecutionPolicy exec, const size_t n)
{
  thrust::host_vector<T> h_data   = unittest::random_integers<T>(n);
  thrust::device_vector<T> d_data = h_data;

  thrust::device_vector<T> d_result(1);

  T init = 13;

  T h_result = thrust::reduce(h_data.begin(), h_data.end(), init);

  reduce_kernel<<<1, 1>>>(exec, d_data.begin(), d_data.end(), init, d_result.begin());
  cudaError_t const err = cudaDeviceSynchronize();
  REQUIRE(cudaSuccess == err);

  REQUIRE(h_result == d_result[0]);
}

template <typename T>
struct TestReduceDeviceSeq
{
  void operator()(const size_t n)
  {
    TestReduceDevice<T>(thrust::seq, n);
  }
};
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES(TestReduceDeviceSeq, IntegralTypes);

template <typename T>
struct TestReduceDeviceDevice
{
  void operator()(const size_t n)
  {
    TestReduceDevice<T>(thrust::device, n);
  }
};
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES(TestReduceDeviceDevice, IntegralTypes);

template <typename T>
struct TestReduceDeviceNoSync
{
  void operator()(const size_t n)
  {
    TestReduceDevice<T>(thrust::cuda::par_nosync, n);
  }
};
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES(TestReduceDeviceNoSync, IntegralTypes);
#endif

template <typename ExecutionPolicy>
void test_reduce_cuda_streams(ExecutionPolicy policy)
{
  using Vector = thrust::device_vector<int>;

  Vector v(3);
  v[0] = 1;
  v[1] = -2;
  v[2] = 3;

  cudaStream_t s;
  cudaStreamCreate(&s);

  auto streampolicy = policy.on(s);

  // no initializer
  REQUIRE(thrust::reduce(streampolicy, v.begin(), v.end()) == 2);

  // with initializer
  REQUIRE(thrust::reduce(streampolicy, v.begin(), v.end(), 10) == 12);

  cudaStreamDestroy(s);
}

TEST_CASE("TestReduceCudaStreamsSync", "[reduce]")
{
  test_reduce_cuda_streams(thrust::cuda::par);
}

TEST_CASE("TestReduceCudaStreamsNoSync", "[reduce]")
{
  test_reduce_cuda_streams(thrust::cuda::par_nosync);
}

#if defined(THRUST_RDC_ENABLED)
TEST_CASE("TestReduceLargeInput", "[reduce]")
{
  using T                 = unsigned long long;
  using OffsetT           = std::size_t;
  const OffsetT num_items = 1ull << 32;

  const cuda::constant_iterator<T> d_data(T{1});
  thrust::device_vector<T> d_result(1);

  reduce_kernel<<<1, 1>>>(thrust::device, d_data, d_data + num_items, T{}, d_result.begin());
  cudaError_t const err = cudaDeviceSynchronize();
  REQUIRE(cudaSuccess == err);

  REQUIRE(num_items == d_result[0]);
}
#endif
