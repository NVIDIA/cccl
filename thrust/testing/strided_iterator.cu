#include <cuda/__cccl_config>

_CCCL_DIAG_PUSH
// gcc 10 and 11 wrongly warn about an out-of-bounds access in TestWritingStridedIteratorToStructMember
#if _CCCL_COMPILER(GCC, >=, 10) && _CCCL_COMPILER(GCC, <, 12)
_CCCL_DIAG_SUPPRESS_GCC("-Warray-bounds")
#endif // _CCCL_COMPILER(GCC, >=, 10) && _CCCL_COMPILER(GCC, <, 12)

#include <thrust/device_vector.h>
#include <thrust/iterator/strided_iterator.h>

#include <cuda/std/array>
#include <cuda/std/utility>

#include <algorithm>
#include <numeric>

#include <unittest/unittest.h>

void TestReadingStridedIterator()
{
  thrust::host_vector<int> v(21);
  std::iota(v.begin(), v.end(), -4);
  auto iter = thrust::make_strided_iterator(v.begin() + 4, 2);

  ASSERT_EQUAL(*iter, 0);
  iter++;
  ASSERT_EQUAL(*iter, 2);
  iter++;
  iter++;
  ASSERT_EQUAL(*iter, 6);
  iter += 5;
  ASSERT_EQUAL(*iter, 16);
  iter -= 10;
  ASSERT_EQUAL(*iter, -4);
}
DECLARE_UNITTEST(TestReadingStridedIterator);

template <typename Vector>
void TestWritingStridedIterator()
{
  // iterate over all second elements (runtime stride)
  {
    Vector v(10);
    auto iter = thrust::make_strided_iterator(v.begin(), 2);
    ASSERT_EQUAL(v, (Vector{0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
    *iter = 33;
    ASSERT_EQUAL(v, (Vector{33, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
    auto iter2 = iter + 1;
    *iter2     = 34;
    ASSERT_EQUAL(v, (Vector{33, 0, 34, 0, 0, 0, 0, 0, 0, 0}));
    thrust::fill(iter + 2, iter + 4, 42);
    ASSERT_EQUAL(v, (Vector{33, 0, 34, 0, 42, 0, 42, 0, 0, 0}));
  }

  // iterate over all second elements (static stride)
  {
    Vector v(10);
    auto iter = thrust::make_strided_iterator<2>(v.begin());
    thrust::fill(iter, iter + 3, 42);
    ASSERT_EQUAL(v, (Vector{42, 0, 42, 0, 42, 0, 0, 0, 0, 0}));
  }
}
DECLARE_INTEGRAL_VECTOR_UNITTEST(TestWritingStridedIterator);

void TestWritingStridedIteratorToStructMember()
{
  using pair            = ::cuda::std::pair<int, double>;
  using arr_of_pairs    = ::cuda::std::array<pair, 4>;
  const auto data       = arr_of_pairs{{{1, 2}, {3, 4}, {5, 6}, {7, 8}}};
  const auto reference  = arr_of_pairs{{{1, 1337}, {3, 1337}, {5, 1337}, {7, 1337}}};
  constexpr auto stride = sizeof(pair) / sizeof(double);
  static_assert(stride == 2);

  // iterate over all second elements (runtime stride)
  {
    auto arr  = data;
    auto iter = thrust::make_strided_iterator(&arr[0].second, stride);
    thrust::fill(iter, iter + 4, 1337);
    ASSERT_EQUAL(arr == reference, true);
  }

  // iterate over all second elements (static stride)
  {
    auto arr  = data;
    auto iter = thrust::make_strided_iterator<stride>(&arr[0].second);
    thrust::fill(iter, iter + 4, 1337);
    ASSERT_EQUAL(arr == reference, true);
  }
}
DECLARE_UNITTEST(TestWritingStridedIteratorToStructMember);

_CCCL_DIAG_POP
