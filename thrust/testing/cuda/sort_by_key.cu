#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/sort.h>

#include <unittest/unittest.h>

template <typename T>
struct my_less
{
  _CCCL_HOST_DEVICE bool operator()(const T& lhs, const T& rhs) const
  {
    return lhs < rhs;
  }
};

#ifdef THRUST_TEST_DEVICE_SIDE
template <typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Compare>
__global__ void sort_by_key_kernel(
  ExecutionPolicy exec, Iterator1 keys_first, Iterator1 keys_last, Iterator2 values_first, Compare comp)
{
  thrust::sort_by_key(exec, keys_first, keys_last, values_first, comp);
}

template <typename T, typename ExecutionPolicy, typename Compare>
void TestComparisonSortByKeyDevice(ExecutionPolicy exec, const size_t n, Compare comp)
{
  thrust::host_vector<T> h_keys   = unittest::random_integers<T>(n);
  thrust::device_vector<T> d_keys = h_keys;

  thrust::host_vector<T> h_values   = h_keys;
  thrust::device_vector<T> d_values = d_keys;

  sort_by_key_kernel<<<1, 1>>>(exec, d_keys.begin(), d_keys.end(), d_values.begin(), comp);
  cudaError_t const err = cudaDeviceSynchronize();
  ASSERT_EQUAL(cudaSuccess, err);

  thrust::sort_by_key(h_keys.begin(), h_keys.end(), h_values.begin(), comp);

  ASSERT_EQUAL(h_keys, d_keys);
  ASSERT_EQUAL(h_values, d_values);
};

template <typename T>
struct TestComparisonSortByKeyDeviceSeq
{
  void operator()(const size_t n)
  {
    TestComparisonSortByKeyDevice<T>(thrust::seq, n, my_less<T>());
  }
};
VariableUnitTest<TestComparisonSortByKeyDeviceSeq, unittest::type_list<unittest::int8_t, unittest::int32_t>>
  TestComparisonSortByKeyDeviceSeqInstance;

template <typename T>
struct TestComparisonSortByKeyDeviceDevice
{
  void operator()(const size_t n)
  {
    TestComparisonSortByKeyDevice<T>(thrust::device, n, my_less<T>());
  }
};
VariableUnitTest<TestComparisonSortByKeyDeviceDevice, unittest::type_list<unittest::int8_t, unittest::int32_t>>
  TestComparisonSortByKeyDeviceDeviceDeviceInstance;

template <typename T, typename ExecutionPolicy>
void TestSortByKeyDevice(ExecutionPolicy exec, const size_t n)
{
  TestComparisonSortByKeyDevice<T>(exec, n, thrust::less<T>());
};

template <typename T>
struct TestSortByKeyDeviceSeq
{
  void operator()(const size_t n)
  {
    TestSortByKeyDevice<T>(thrust::seq, n);
  }
};
VariableUnitTest<TestSortByKeyDeviceSeq, unittest::type_list<unittest::int8_t, unittest::int32_t>>
  TestSortByKeyDeviceSeqInstance;

template <typename T>
struct TestSortByKeyDeviceDevice
{
  void operator()(const size_t n)
  {
    TestSortByKeyDevice<T>(thrust::device, n);
  }
};
VariableUnitTest<TestSortByKeyDeviceDevice, unittest::type_list<unittest::int8_t, unittest::int32_t>>
  TestSortByKeyDeviceDeviceInstance;
#endif

void TestComparisonSortByKeyCudaStreams()
{
  thrust::device_vector<int> keys{9, 3, 2, 0, 4, 7, 8, 1, 5, 6};
  thrust::device_vector<int> vals{9, 3, 2, 0, 4, 7, 8, 1, 5, 6};

  cudaStream_t s;
  cudaStreamCreate(&s);

  thrust::sort_by_key(thrust::cuda::par.on(s), keys.begin(), keys.end(), vals.begin(), my_less<int>());
  cudaStreamSynchronize(s);

  ASSERT_EQUAL(true, thrust::is_sorted(keys.begin(), keys.end()));
  ASSERT_EQUAL(true, thrust::is_sorted(vals.begin(), vals.end()));

  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestComparisonSortByKeyCudaStreams);

void TestSortByKeyCudaStreams()
{
  thrust::device_vector<int> keys{9, 3, 2, 0, 4, 7, 8, 1, 5, 6};
  thrust::device_vector<int> vals{9, 3, 2, 0, 4, 7, 8, 1, 5, 6};

  cudaStream_t s;
  cudaStreamCreate(&s);

  thrust::sort_by_key(thrust::cuda::par.on(s), keys.begin(), keys.end(), vals.begin());
  cudaStreamSynchronize(s);

  ASSERT_EQUAL(true, thrust::is_sorted(keys.begin(), keys.end()));
  ASSERT_EQUAL(true, thrust::is_sorted(vals.begin(), vals.end()));

  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestSortByKeyCudaStreams);
