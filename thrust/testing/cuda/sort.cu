#include <thrust/copy.h>
#include <thrust/equal.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/reverse_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/sort.h>

#include <cuda/std/limits>

#include <algorithm>
#include <cstdint>
#include <exception>

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
template <typename ExecutionPolicy, typename Iterator, typename Compare>
__global__ void sort_kernel(ExecutionPolicy exec, Iterator first, Iterator last, Compare comp)
{
  thrust::sort(exec, first, last, comp);
}

template <typename T, typename ExecutionPolicy, typename Compare>
void TestComparisonSortDevice(ExecutionPolicy exec, const size_t n, Compare comp)
{
  thrust::host_vector<T> h_data   = unittest::random_integers<T>(n);
  thrust::device_vector<T> d_data = h_data;

  sort_kernel<<<1, 1>>>(exec, d_data.begin(), d_data.end(), comp);
  cudaError_t const err = cudaDeviceSynchronize();
  ASSERT_EQUAL(cudaSuccess, err);

  thrust::sort(h_data.begin(), h_data.end(), comp);

  ASSERT_EQUAL(h_data, d_data);
};

template <typename T>
struct TestComparisonSortDeviceSeq
{
  void operator()(const size_t n)
  {
    TestComparisonSortDevice<T>(thrust::seq, n, my_less<T>());
  }
};
VariableUnitTest<TestComparisonSortDeviceSeq, unittest::type_list<unittest::int8_t, unittest::int32_t>>
  TestComparisonSortDeviceSeqInstance;

template <typename T>
struct TestComparisonSortDeviceDevice
{
  void operator()(const size_t n)
  {
    TestComparisonSortDevice<T>(thrust::device, n, my_less<T>());
  }
};
VariableUnitTest<TestComparisonSortDeviceDevice, unittest::type_list<unittest::int8_t, unittest::int32_t>>
  TestComparisonSortDeviceDeviceDeviceInstance;

template <typename T, typename ExecutionPolicy>
void TestSortDevice(ExecutionPolicy exec, const size_t n)
{
  TestComparisonSortDevice<T>(exec, n, thrust::less<T>());
};

template <typename T>
struct TestSortDeviceSeq
{
  void operator()(const size_t n)
  {
    TestSortDevice<T>(thrust::seq, n);
  }
};
VariableUnitTest<TestSortDeviceSeq, unittest::type_list<unittest::int8_t, unittest::int32_t>> TestSortDeviceSeqInstance;

template <typename T>
struct TestSortDeviceDevice
{
  void operator()(const size_t n)
  {
    TestSortDevice<T>(thrust::device, n);
  }
};
VariableUnitTest<TestSortDeviceDevice, unittest::type_list<unittest::int8_t, unittest::int32_t>>
  TestSortDeviceDeviceInstance;
#endif

void TestSortCudaStreams()
{
  thrust::device_vector<int> keys{9, 3, 2, 0, 4, 7, 8, 1, 5, 6};

  cudaStream_t s;
  cudaStreamCreate(&s);

  thrust::sort(thrust::cuda::par.on(s), keys.begin(), keys.end());
  cudaStreamSynchronize(s);

  ASSERT_EQUAL(true, thrust::is_sorted(keys.begin(), keys.end()));

  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestSortCudaStreams);

void TestComparisonSortCudaStreams()
{
  thrust::device_vector<int> keys{9, 3, 2, 0, 4, 7, 8, 1, 5, 6};

  cudaStream_t s;
  cudaStreamCreate(&s);

  thrust::sort(thrust::cuda::par.on(s), keys.begin(), keys.end(), my_less<int>());
  cudaStreamSynchronize(s);

  ASSERT_EQUAL(true, thrust::is_sorted(keys.begin(), keys.end(), my_less<int>()));

  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestComparisonSortCudaStreams);

template <typename T>
struct TestRadixSortDispatch
{
  static_assert(thrust::cuda_cub::__smart_sort::can_use_primitive_sort<T, thrust::less<T>>::value, "");
  static_assert(thrust::cuda_cub::__smart_sort::can_use_primitive_sort<T, thrust::greater<T>>::value, "");
  static_assert(thrust::cuda_cub::__smart_sort::can_use_primitive_sort<T, ::cuda::std::less<T>>::value, "");
  static_assert(thrust::cuda_cub::__smart_sort::can_use_primitive_sort<T, ::cuda::std::greater<T>>::value, "");

  static_assert(thrust::cuda_cub::__smart_sort::can_use_primitive_sort<T, thrust::less<>>::value, "");
  static_assert(thrust::cuda_cub::__smart_sort::can_use_primitive_sort<T, thrust::greater<>>::value, "");
  static_assert(thrust::cuda_cub::__smart_sort::can_use_primitive_sort<T, ::cuda::std::less<>>::value, "");
  static_assert(thrust::cuda_cub::__smart_sort::can_use_primitive_sort<T, ::cuda::std::greater<>>::value, "");

  void operator()() const {}
};
SimpleUnitTest<TestRadixSortDispatch,
               unittest::concat<IntegralTypes,
                                FloatingPointTypes
#ifndef _LIBCUDACXX_HAS_NO_INT128
                                ,
                                unittest::type_list<__int128_t, __uint128_t>
#endif // _LIBCUDACXX_HAS_NO_INT128
#ifdef _CCCL_HAS_NVFP16
                                ,
                                unittest::type_list<__half>
#endif // _CCCL_HAS_NVFP16
#ifdef _CCCL_HAS_NVBF16
                                ,
                                unittest::type_list<__nv_bfloat16>
#endif // _CCCL_HAS_NVBF16
                                >>
  TestRadixSortDispatchInstance;

/**
 * Copy of CUB testing utility
 */
template <typename UnsignedIntegralKeyT>
struct index_to_key_value_op
{
  static constexpr std::size_t max_key_value =
    static_cast<std::size_t>(::cuda::std::numeric_limits<UnsignedIntegralKeyT>::max());
  static constexpr std::size_t lowest_key_value =
    static_cast<std::size_t>(::cuda::std::numeric_limits<UnsignedIntegralKeyT>::lowest());
  static_assert(sizeof(UnsignedIntegralKeyT) < sizeof(std::size_t),
                "Calculation of num_distinct_key_values would overflow");
  static constexpr std::size_t num_distinct_key_values = (max_key_value - lowest_key_value + std::size_t{1ULL});

  __device__ __host__ UnsignedIntegralKeyT operator()(std::size_t index)
  {
    return static_cast<UnsignedIntegralKeyT>(index % num_distinct_key_values);
  }
};

/**
 * Copy of CUB testing utility
 */
template <typename UnsignedIntegralKeyT>
class index_to_expected_key_op
{
private:
  static constexpr std::size_t max_key_value =
    static_cast<std::size_t>(::cuda::std::numeric_limits<UnsignedIntegralKeyT>::max());
  static constexpr std::size_t lowest_key_value =
    static_cast<std::size_t>(::cuda::std::numeric_limits<UnsignedIntegralKeyT>::lowest());
  static_assert(sizeof(UnsignedIntegralKeyT) < sizeof(std::size_t),
                "Calculation of num_distinct_key_values would overflow");
  static constexpr std::size_t num_distinct_key_values = (max_key_value - lowest_key_value + std::size_t{1ULL});

  // item_count / num_distinct_key_values
  std::size_t expected_count_per_item;
  // num remainder items: item_count%num_distinct_key_values
  std::size_t num_remainder_items;
  // remainder item_count: expected_count_per_item+1
  std::size_t remainder_item_count;

public:
  index_to_expected_key_op(std::size_t num_total_items)
      : expected_count_per_item(num_total_items / num_distinct_key_values)
      , num_remainder_items(num_total_items % num_distinct_key_values)
      , remainder_item_count(expected_count_per_item + std::size_t{1ULL})
  {}

  __device__ __host__ UnsignedIntegralKeyT operator()(std::size_t index)
  {
    // The first (num_remainder_items * remainder_item_count) are items that appear once more often than the items that
    // follow remainder_items_offset
    std::size_t remainder_items_offset = num_remainder_items * remainder_item_count;

    UnsignedIntegralKeyT target_item_index =
      (index <= remainder_items_offset)
        ?
        // This is one of the remainder items
        static_cast<UnsignedIntegralKeyT>(index / remainder_item_count)
        :
        // This is an item that appears exactly expected_count_per_item times
        static_cast<UnsignedIntegralKeyT>(
          num_remainder_items + ((index - remainder_items_offset) / expected_count_per_item));
    return target_item_index;
  }
};

void TestSortWithMagnitude(int magnitude)
{
  try
  {
    const std::size_t num_items = 1ull << magnitude;
    thrust::device_vector<std::uint8_t> vec(num_items);
    auto counting_it   = thrust::make_counting_iterator(std::size_t{0});
    auto key_value_it  = thrust::make_transform_iterator(counting_it, index_to_key_value_op<std::uint8_t>{});
    auto rev_sorted_it = thrust::make_reverse_iterator(key_value_it + num_items);
    thrust::copy(rev_sorted_it, rev_sorted_it + num_items, vec.begin());
    thrust::sort(vec.begin(), vec.end());
    auto expected_result_it = thrust::make_transform_iterator(
      thrust::make_counting_iterator(std::size_t{}), index_to_expected_key_op<std::uint8_t>(num_items));
    const bool ok = thrust::equal(expected_result_it, expected_result_it + num_items, vec.cbegin());
    ASSERT_EQUAL(ok, true);
  }
  catch (std::bad_alloc&)
  {}
}

void TestSortWithLargeNumberOfItems()
{
  TestSortWithMagnitude(39);
  TestSortWithMagnitude(32);
  TestSortWithMagnitude(33);
}
DECLARE_UNITTEST(TestSortWithLargeNumberOfItems);

template <typename T>
struct TestSortAscendingKey
{
  void operator()() const
  {
    constexpr int n = 10000;

    thrust::host_vector<T> h_data   = unittest::random_integers<T>(n);
    thrust::device_vector<T> d_data = h_data;

    std::sort(h_data.begin(), h_data.end(), thrust::less<T>{});
    thrust::sort(d_data.begin(), d_data.end(), thrust::less<T>{});

    ASSERT_EQUAL_QUIET(h_data, d_data);
  }
};

SimpleUnitTest<TestSortAscendingKey,
               unittest::concat<unittest::type_list<>
#ifndef _LIBCUDACXX_HAS_NO_INT128
                                ,
                                unittest::type_list<__int128_t, __uint128_t>
#endif
// CTK 12.2 offers __host__ __device__ operators for __half and __nv_bfloat16, so we can use std::sort
#if _CCCL_CUDACC_VER >= 1202000
#  if defined(_CCCL_HAS_NVFP16) || !defined(__CUDA_NO_HALF_OPERATORS__) && !defined(__CUDA_NO_HALF_CONVERSIONS__)
                                ,
                                unittest::type_list<__half>
#  endif
#  if defined(_CCCL_HAS_NVBF16) \
    || !defined(__CUDA_NO_BFLOAT16_OPERATORS__) && !defined(__CUDA_NO_BFLOAT16_CONVERSIONS__)
                                ,
                                unittest::type_list<__nv_bfloat16>
#  endif
#endif // _CCCL_CUDACC_VER >= 1202000
                                >>
  TestSortAscendingKeyMoreTypes;
