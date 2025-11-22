#include <thrust/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>

#include <unittest/unittest.h>

template <typename ExecutionPolicy, typename InputIter, typename OutputIter, typename T>
__global__ void reduce_into_kernel(ExecutionPolicy exec, InputIter first, InputIter last, OutputIter result, T init)
{
  thrust::reduce_into(exec, first, last, result, init);
}

#ifdef THRUST_TEST_DEVICE_SIDE
template <typename T, typename ExecutionPolicy>
void TestReduceIntoDevice(ExecutionPolicy exec, const size_t n)
{
  thrust::host_vector<T> h_data   = unittest::random_integers<T>(n);
  thrust::device_vector<T> d_data = h_data;

  thrust::host_vector<T> h_result(1);
  thrust::device_vector<T> d_result(1);

  T init = 13;

  thrust::reduce_into(h_data.begin(), h_data.end(), h_result.begin(), init);

  reduce_into_kernel<<<1, 1>>>(exec, d_data.begin(), d_data.end(), d_result.begin(), init);
  cudaError_t const err = cudaDeviceSynchronize();
  ASSERT_EQUAL(cudaSuccess, err);

  ASSERT_EQUAL(h_result, d_result);
}

template <typename T>
struct TestReduceIntoDeviceSeq
{
  void operator()(const size_t n)
  {
    TestReduceIntoDevice<T>(thrust::seq, n);
  }
};
VariableUnitTest<TestReduceIntoDeviceSeq, IntegralTypes> TestReduceIntoDeviceSeqInstance;

template <typename T>
struct TestReduceIntoDeviceDevice
{
  void operator()(const size_t n)
  {
    TestReduceIntoDevice<T>(thrust::device, n);
  }
};
VariableUnitTest<TestReduceIntoDeviceDevice, IntegralTypes> TestReduceIntoDeviceDeviceInstance;

template <typename T>
struct TestReduceIntoDeviceNoSync
{
  void operator()(const size_t n)
  {
    TestReduceIntoDevice<T>(thrust::cuda::par_nosync, n);
  }
};
VariableUnitTest<TestReduceIntoDeviceNoSync, IntegralTypes> TestReduceIntoDeviceNoSyncInstance;
#endif

template <typename ExecutionPolicy>
void TestReduceIntoCudaStreams(ExecutionPolicy policy)
{
  using Vector = thrust::device_vector<int>;

  Vector v = {1, -2, 3};

  Vector o(1);

  cudaStream_t s;
  cudaStreamCreate(&s);

  auto streampolicy = policy.on(s);

  // no initializer
  thrust::reduce_into(streampolicy, v.begin(), v.end(), o.begin());

  cudaStreamSynchronize(s);
  ASSERT_EQUAL(o[0], 2);

  // with initializer
  thrust::reduce_into(streampolicy, v.begin(), v.end(), o.begin(), 10);

  cudaStreamSynchronize(s);
  ASSERT_EQUAL(o[0], 12);

  cudaStreamDestroy(s);
}

void TestReduceIntoCudaStreamsSync()
{
  TestReduceIntoCudaStreams(thrust::cuda::par);
}
DECLARE_UNITTEST(TestReduceIntoCudaStreamsSync);

void TestReduceIntoCudaStreamsNoSync()
{
  TestReduceIntoCudaStreams(thrust::cuda::par_nosync);
}
DECLARE_UNITTEST(TestReduceIntoCudaStreamsNoSync);

#if defined(THRUST_RDC_ENABLED)
void TestReduceIntoLargeInput()
{
  using T                 = unsigned long long;
  using OffsetT           = std::size_t;
  const OffsetT num_items = 1ull << 32;

  thrust::constant_iterator<T> d_data(T{1});
  thrust::device_vector<T> d_result(1);

  reduce_into_kernel<<<1, 1>>>(thrust::device, d_data, d_data + num_items, d_result.begin(), T{});
  cudaError_t const err = cudaDeviceSynchronize();
  ASSERT_EQUAL(cudaSuccess, err);

  ASSERT_EQUAL(num_items, d_result[0]);
}
DECLARE_UNITTEST(TestReduceIntoLargeInput);
#endif
