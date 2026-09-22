#include <thrust/binary_search.h>
#include <thrust/distance.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sort.h>

#include <cuda/std/algorithm>
#include <cuda/std/iterator>
#include <cuda/std/type_traits>

#include <complex>
#include <cstdint>
#include <numeric>

#include <unittest/unittest.h>

template <typename ValueType, typename DifferenceType>
inline constexpr bool diff_type_is =
  ::cuda::std::is_same_v<typename thrust::counting_iterator<ValueType>::difference_type, DifferenceType>;

static_assert(diff_type_is<int8_t, int>);
static_assert(diff_type_is<uint8_t, int>);
static_assert(diff_type_is<int16_t, int>);
static_assert(diff_type_is<uint16_t, int>);
static_assert(diff_type_is<int32_t, ptrdiff_t>);
static_assert(diff_type_is<uint32_t, ptrdiff_t>);
static_assert(diff_type_is<int64_t, ptrdiff_t>);
static_assert(diff_type_is<uint64_t, ptrdiff_t>);
#if _CCCL_HAS_INT128()
static_assert(diff_type_is<__int128_t, ptrdiff_t>);
static_assert(diff_type_is<__uint128_t, ptrdiff_t>);
#endif
static_assert(diff_type_is<float, ptrdiff_t>);
static_assert(diff_type_is<double, ptrdiff_t>);

struct custom_int
{
  _CCCL_HOST_DEVICE custom_int(int) {}
  _CCCL_HOST_DEVICE operator int() const;
};
static_assert(diff_type_is<custom_int, ptrdiff_t>);

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_MSVC(4244 4267) // possible loss of data

// ensure that we properly support thrust::counting_iterator from cuda::std
TEST_CASE("TestCountingIteratorTraits", "[counting_iterator]")
{
  using it       = thrust::counting_iterator<int>;
  using traits   = cuda::std::iterator_traits<it>;
  using category = thrust::detail::iterator_category_with_system_and_traversal<::cuda::std::random_access_iterator_tag,
                                                                               thrust::any_system_tag,
                                                                               thrust::random_access_traversal_tag>;

  static_assert(cuda::std::is_same_v<traits::difference_type, ptrdiff_t>);
  static_assert(cuda::std::is_same_v<traits::value_type, int>);
  static_assert(cuda::std::is_same_v<traits::pointer, void>);
  static_assert(cuda::std::is_same_v<traits::reference, signed int>);
  static_assert(cuda::std::is_same_v<traits::iterator_category, category>);

  static_assert(cuda::std::is_same_v<thrust::iterator_traversal_t<it>, thrust::random_access_traversal_tag>);

  static_assert(cuda::std::__has_random_access_traversal<it>);

  static_assert(!cuda::std::output_iterator<it, int>);
  static_assert(cuda::std::input_iterator<it>);
  static_assert(cuda::std::forward_iterator<it>);
  static_assert(cuda::std::bidirectional_iterator<it>);
  static_assert(cuda::std::random_access_iterator<it>);
  static_assert(!cuda::std::contiguous_iterator<it>);
}

template <typename T>
void TestCountingDefaultConstructor()
{
  const thrust::counting_iterator<T> iter0;
  REQUIRE(*iter0 == T{});
}
DECLARE_GENERIC_UNITTEST(TestCountingDefaultConstructor);

TEST_CASE("TestCountingIteratorCopyConstructor", "[counting_iterator]")
{
  const thrust::counting_iterator<int> iter0(100);

  const thrust::counting_iterator<int> iter1(iter0);

  REQUIRE(iter0 == iter1);
  REQUIRE(*iter0 == *iter1);

  // construct from related space
  const thrust::counting_iterator<int, thrust::host_system_tag> h_iter = iter0;
  REQUIRE(*iter0 == *h_iter);

  const thrust::counting_iterator<int, thrust::device_system_tag> d_iter = iter0;
  REQUIRE(*iter0 == *d_iter);
}
static_assert(cuda::std::is_trivially_copy_constructible<thrust::counting_iterator<int>>::value);
static_assert(cuda::std::is_trivially_copyable<thrust::counting_iterator<int>>::value);

TEST_CASE("TestCountingIteratorIncrement", "[counting_iterator]")
{
  thrust::counting_iterator<int> iter(0);

  REQUIRE(*iter == 0);

  iter++;

  REQUIRE(*iter == 1);

  iter++;
  iter++;

  REQUIRE(*iter == 3);

  iter += 5;

  REQUIRE(*iter == 8);

  iter -= 10;

  REQUIRE(*iter == -2);
}

TEST_CASE("TestCountingIteratorComparison", "[counting_iterator]")
{
  thrust::counting_iterator<int> iter1(0);
  thrust::counting_iterator<int> iter2(0);

  REQUIRE(iter1 - iter2 == 0);
  REQUIRE(iter1 == iter2);

  iter1++;

  REQUIRE(iter1 - iter2 == 1);
  REQUIRE_FALSE(iter1 == iter2);

  iter2++;

  REQUIRE(iter1 - iter2 == 0);
  REQUIRE(iter1 == iter2);

  iter1 += 100;
  iter2 += 100;

  REQUIRE(iter1 - iter2 == 0);
  REQUIRE(iter1 == iter2);
}

TEST_CASE("TestCountingIteratorFloatComparison", "[counting_iterator]")
{
  thrust::counting_iterator<float> iter1(0);
  thrust::counting_iterator<float> iter2(0);

  REQUIRE(iter1 - iter2 == 0);
  REQUIRE(iter1 == iter2);
  REQUIRE_FALSE(iter1 < iter2);
  REQUIRE_FALSE(iter2 < iter1);

  iter1++;

  REQUIRE(iter1 - iter2 == 1);
  REQUIRE_FALSE(iter1 == iter2);
  REQUIRE(iter2 < iter1);
  REQUIRE_FALSE(iter1 < iter2);

  iter2++;

  REQUIRE(iter1 - iter2 == 0);
  REQUIRE(iter1 == iter2);
  REQUIRE_FALSE(iter1 < iter2);
  REQUIRE_FALSE(iter2 < iter1);

  iter1 += 100;
  iter2 += 100;

  REQUIRE(iter1 - iter2 == 0);
  REQUIRE(iter1 == iter2);
  REQUIRE_FALSE(iter1 < iter2);
  REQUIRE_FALSE(iter2 < iter1);

  thrust::counting_iterator<float> iter3(0);
  thrust::counting_iterator<float> iter4(0.5);

  REQUIRE(iter3 - iter4 == 0);
  REQUIRE(iter3 == iter4);
  REQUIRE_FALSE(iter3 < iter4);
  REQUIRE_FALSE(iter4 < iter3);

  iter3++; // iter3 = 1.0, iter4 = 0.5

  REQUIRE(iter3 - iter4 == 0);
  REQUIRE(iter3 == iter4);
  REQUIRE_FALSE(iter3 < iter4);
  REQUIRE_FALSE(iter4 < iter3);

  iter4++; // iter3 = 1.0, iter4 = 1.5

  REQUIRE(iter3 - iter4 == 0);
  REQUIRE(iter3 == iter4);
  REQUIRE_FALSE(iter3 < iter4);
  REQUIRE_FALSE(iter4 < iter3);

  iter4++; // iter3 = 1.0, iter4 = 2.5

  REQUIRE(iter3 - iter4 == -1);
  REQUIRE(iter4 - iter3 == 1);
  REQUIRE_FALSE(iter3 == iter4);
  REQUIRE(iter3 < iter4);
  REQUIRE_FALSE(iter4 < iter3);
}

TEST_CASE("TestCountingIteratorDistance", "[counting_iterator]")
{
  thrust::counting_iterator<int> iter1(0);
  thrust::counting_iterator<int> iter2(5);

  REQUIRE(::cuda::std::distance(iter1, iter2) == 5);

  iter1++;

  REQUIRE(::cuda::std::distance(iter1, iter2) == 4);

  iter2 += 100;

  REQUIRE(::cuda::std::distance(iter1, iter2) == 104);
}

TEST_CASE("TestCountingIteratorUnsignedType", "[counting_iterator]")
{
  const thrust::counting_iterator<unsigned int> iter0(0);
  const thrust::counting_iterator<unsigned int> iter1(5);

  REQUIRE(iter1 - iter0 == 5);
  REQUIRE(iter0 - iter1 == -5);
  REQUIRE(iter0 != iter1);
  REQUIRE(iter0 < iter1);
  REQUIRE_FALSE(iter1 < iter0);
}

TEST_CASE("TestCountingIteratorLowerBound", "[counting_iterator]")
{
  const size_t n = 10000;
  const size_t M = 100;

  thrust::host_vector<unsigned int> h_data = unittest::random_integers<unsigned int>(n);
  for (unsigned int i = 0; i < n; ++i)
  {
    h_data[i] %= M;
  }

  thrust::sort(h_data.begin(), h_data.end());

  thrust::device_vector<unsigned int> d_data = h_data;

  const thrust::counting_iterator<unsigned int> search_begin(0);
  const thrust::counting_iterator<unsigned int> search_end(M);

  thrust::host_vector<unsigned int> h_result(M);
  thrust::device_vector<unsigned int> d_result(M);

  thrust::lower_bound(h_data.begin(), h_data.end(), search_begin, search_end, h_result.begin());

  thrust::lower_bound(d_data.begin(), d_data.end(), search_begin, search_end, d_result.begin());

  REQUIRE(h_result == d_result);
}

TEST_CASE("TestCountingIteratorDifference", "[counting_iterator]")
{
  using Iterator   = thrust::counting_iterator<std::uint64_t>;
  using Difference = thrust::detail::it_difference_t<Iterator>;

  const Difference diff = std::numeric_limits<std::uint32_t>::max() + 1; // NOLINT(bugprone-misplaced-widening-cast)

  const Iterator first(0);
  const Iterator last = first + diff;

  REQUIRE(diff == last - first);
}

TEST_CASE("TestCountingIteratorDynamicStride", "[counting_iterator]")
{
  auto iter = thrust::make_counting_iterator(0, 2);
  static_assert(sizeof(iter) == 2 * sizeof(int));

  REQUIRE(*iter == 0);
  iter++;
  REQUIRE(*iter == 2);
  iter++;
  iter++;
  REQUIRE(*iter == 6);
  iter += 5;
  REQUIRE(*iter == 16);
  iter -= 10;
  REQUIRE(*iter == -4);
}

TEST_CASE("TestCountingIteratorStaticStride", "[counting_iterator]")
{
  auto iter = thrust::make_counting_iterator<2>(0);
  static_assert(sizeof(decltype(iter)) == sizeof(int));

  REQUIRE(*iter == 0);
  iter++;
  REQUIRE(*iter == 2);
  iter++;
  iter++;
  REQUIRE(*iter == 6);
  iter += 5;
  REQUIRE(*iter == 16);
  iter -= 10;
  REQUIRE(*iter == -4);
}

TEST_CASE("TestCountingIteratorPointer", "[counting_iterator]")
{
  int arr[11];
  std::iota(arr, arr + 11, 0);

  auto iter = thrust::make_counting_iterator(&arr[2]);

  REQUIRE(*iter == &arr[2]);
  REQUIRE(**iter == 2);
  iter++;
  REQUIRE(*iter == &arr[3]);
  REQUIRE(**iter == 3);
  iter++;
  iter++;
  REQUIRE(*iter == &arr[5]);
  REQUIRE(**iter == 5);
  iter += 5;
  REQUIRE(*iter == &arr[10]);
  REQUIRE(**iter == 10);
  iter -= 10;
  REQUIRE(*iter == &arr[0]);
  REQUIRE(**iter == 0);
}

_CCCL_DIAG_POP

// Test that counting_iterator<float> distance_to does not trigger
// MSVC C4244 (implicit float-to-integer conversion) without suppression.
TEST_CASE("TestCountingIteratorFloatDistanceTo", "[counting_iterator]")
{
  const thrust::counting_iterator<float> iter1(0);
  const thrust::counting_iterator<float> iter2(5);

  REQUIRE(iter2 - iter1 == 5);
}
