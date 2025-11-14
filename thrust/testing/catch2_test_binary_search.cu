#include <thrust/binary_search.h>
#include <thrust/iterator/retag.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <cuda/std/cstdint>

#include "catch2_test_helper.h"
#include "unittest/special_types.h"

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_MSVC(4244 4267) // possible loss of data

//////////////////////
// Scalar Functions //
//////////////////////

TEMPLATE_LIST_TEST_CASE("ScalarLowerBoundSimple", "[binary_search]", vector_list)
{
  using Vector = TestType;
  Vector vec{0, 2, 5, 7, 8};

  CHECK(thrust::lower_bound(vec.begin(), vec.end(), 0) - vec.begin() == 0);
  CHECK(thrust::lower_bound(vec.begin(), vec.end(), 1) - vec.begin() == 1);
  CHECK(thrust::lower_bound(vec.begin(), vec.end(), 2) - vec.begin() == 1);
  CHECK(thrust::lower_bound(vec.begin(), vec.end(), 3) - vec.begin() == 2);
  CHECK(thrust::lower_bound(vec.begin(), vec.end(), 4) - vec.begin() == 2);
  CHECK(thrust::lower_bound(vec.begin(), vec.end(), 5) - vec.begin() == 2);
  CHECK(thrust::lower_bound(vec.begin(), vec.end(), 6) - vec.begin() == 3);
  CHECK(thrust::lower_bound(vec.begin(), vec.end(), 7) - vec.begin() == 3);
  CHECK(thrust::lower_bound(vec.begin(), vec.end(), 8) - vec.begin() == 4);
  CHECK(thrust::lower_bound(vec.begin(), vec.end(), 9) - vec.begin() == 5);
}

template <typename ForwardIterator, typename LessThanComparable>
ForwardIterator
lower_bound(my_system& system, ForwardIterator first, ForwardIterator /*last*/, const LessThanComparable& /*value*/)
{
  system.validate_dispatch();
  return first;
}

TEST_CASE("ScalarLowerBoundDispatchExplicit", "[binary_search]")
{
  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::lower_bound(sys, vec.begin(), vec.end(), 0);

  CHECK(sys.is_valid());
}

template <typename ForwardIterator, typename LessThanComparable>
ForwardIterator lower_bound(my_tag, ForwardIterator first, ForwardIterator /*last*/, const LessThanComparable& /*value*/)
{
  *first = 13;
  return first;
}

TEST_CASE("ScalarLowerBoundDispatchImplicit", "[binary_search]")
{
  thrust::device_vector<int> vec(1);

  thrust::lower_bound(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.end()), 0);

  CHECK(13 == vec.front());
}

TEMPLATE_LIST_TEST_CASE("ScalarUpperBoundSimple", "[binary_search]", vector_list)
{
  using Vector = TestType;
  Vector vec{0, 2, 5, 7, 8};

  CHECK(thrust::upper_bound(vec.begin(), vec.end(), 0) - vec.begin() == 1);
  CHECK(thrust::upper_bound(vec.begin(), vec.end(), 1) - vec.begin() == 1);
  CHECK(thrust::upper_bound(vec.begin(), vec.end(), 2) - vec.begin() == 2);
  CHECK(thrust::upper_bound(vec.begin(), vec.end(), 3) - vec.begin() == 2);
  CHECK(thrust::upper_bound(vec.begin(), vec.end(), 4) - vec.begin() == 2);
  CHECK(thrust::upper_bound(vec.begin(), vec.end(), 5) - vec.begin() == 3);
  CHECK(thrust::upper_bound(vec.begin(), vec.end(), 6) - vec.begin() == 3);
  CHECK(thrust::upper_bound(vec.begin(), vec.end(), 7) - vec.begin() == 4);
  CHECK(thrust::upper_bound(vec.begin(), vec.end(), 8) - vec.begin() == 5);
  CHECK(thrust::upper_bound(vec.begin(), vec.end(), 9) - vec.begin() == 5);
}

template <typename ForwardIterator, typename LessThanComparable>
ForwardIterator
upper_bound(my_system& system, ForwardIterator first, ForwardIterator /*last*/, const LessThanComparable& /*value*/)
{
  system.validate_dispatch();
  return first;
}

TEST_CASE("ScalarUpperBoundDispatchExplicit", "[binary_search]")
{
  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::upper_bound(sys, vec.begin(), vec.end(), 0);

  CHECK(sys.is_valid());
}

template <typename ForwardIterator, typename LessThanComparable>
ForwardIterator upper_bound(my_tag, ForwardIterator first, ForwardIterator /*last*/, const LessThanComparable& /*value*/)
{
  *first = 13;
  return first;
}

TEST_CASE("ScalarUpperBoundDispatchImplicit", "[binary_search]")
{
  thrust::device_vector<int> vec(1);

  thrust::upper_bound(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.end()), 0);

  CHECK(13 == vec.front());
}

TEMPLATE_LIST_TEST_CASE("ScalarBinarySearchSimple", "[binary_search]", vector_list)
{
  using Vector = TestType;
  Vector vec{0, 2, 5, 7, 8};

  CHECK(thrust::binary_search(vec.begin(), vec.end(), 0) == true);
  CHECK(thrust::binary_search(vec.begin(), vec.end(), 1) == false);
  CHECK(thrust::binary_search(vec.begin(), vec.end(), 2) == true);
  CHECK(thrust::binary_search(vec.begin(), vec.end(), 3) == false);
  CHECK(thrust::binary_search(vec.begin(), vec.end(), 4) == false);
  CHECK(thrust::binary_search(vec.begin(), vec.end(), 5) == true);
  CHECK(thrust::binary_search(vec.begin(), vec.end(), 6) == false);
  CHECK(thrust::binary_search(vec.begin(), vec.end(), 7) == true);
  CHECK(thrust::binary_search(vec.begin(), vec.end(), 8) == true);
  CHECK(thrust::binary_search(vec.begin(), vec.end(), 9) == false);
}

template <typename ForwardIterator, typename LessThanComparable>
bool binary_search(
  my_system& system, ForwardIterator /*first*/, ForwardIterator /*last*/, const LessThanComparable& /*value*/)
{
  system.validate_dispatch();
  return false;
}

TEST_CASE("ScalarBinarySearchDispatchExplicit", "[binary_search]")
{
  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::binary_search(sys, vec.begin(), vec.end(), 0);

  CHECK(sys.is_valid());
}

template <typename ForwardIterator, typename LessThanComparable>
bool binary_search(my_tag, ForwardIterator first, ForwardIterator /*last*/, const LessThanComparable& /*value*/)
{
  *first = 13;
  return false;
}

TEST_CASE("ScalarBinarySearchDispatchImplicit", "[binary_search]")
{
  thrust::device_vector<int> vec(1);

  thrust::binary_search(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.end()), 0);

  CHECK(13 == vec.front());
}

TEMPLATE_LIST_TEST_CASE("ScalarEqualRangeSimple", "[binary_search]", vector_list)
{
  using Vector = TestType;
  Vector vec{0, 2, 5, 7, 8};

  CHECK(thrust::equal_range(vec.begin(), vec.end(), 0).first - vec.begin() == 0);
  CHECK(thrust::equal_range(vec.begin(), vec.end(), 1).first - vec.begin() == 1);
  CHECK(thrust::equal_range(vec.begin(), vec.end(), 2).first - vec.begin() == 1);
  CHECK(thrust::equal_range(vec.begin(), vec.end(), 3).first - vec.begin() == 2);
  CHECK(thrust::equal_range(vec.begin(), vec.end(), 4).first - vec.begin() == 2);
  CHECK(thrust::equal_range(vec.begin(), vec.end(), 5).first - vec.begin() == 2);
  CHECK(thrust::equal_range(vec.begin(), vec.end(), 6).first - vec.begin() == 3);
  CHECK(thrust::equal_range(vec.begin(), vec.end(), 7).first - vec.begin() == 3);
  CHECK(thrust::equal_range(vec.begin(), vec.end(), 8).first - vec.begin() == 4);
  CHECK(thrust::equal_range(vec.begin(), vec.end(), 9).first - vec.begin() == 5);

  CHECK(thrust::equal_range(vec.begin(), vec.end(), 0).second - vec.begin() == 1);
  CHECK(thrust::equal_range(vec.begin(), vec.end(), 1).second - vec.begin() == 1);
  CHECK(thrust::equal_range(vec.begin(), vec.end(), 2).second - vec.begin() == 2);
  CHECK(thrust::equal_range(vec.begin(), vec.end(), 3).second - vec.begin() == 2);
  CHECK(thrust::equal_range(vec.begin(), vec.end(), 4).second - vec.begin() == 2);
  CHECK(thrust::equal_range(vec.begin(), vec.end(), 5).second - vec.begin() == 3);
  CHECK(thrust::equal_range(vec.begin(), vec.end(), 6).second - vec.begin() == 3);
  CHECK(thrust::equal_range(vec.begin(), vec.end(), 7).second - vec.begin() == 4);
  CHECK(thrust::equal_range(vec.begin(), vec.end(), 8).second - vec.begin() == 5);
  CHECK(thrust::equal_range(vec.begin(), vec.end(), 9).second - vec.begin() == 5);
}

template <typename ForwardIterator, typename LessThanComparable>
cuda::std::pair<ForwardIterator, ForwardIterator>
equal_range(my_system& system, ForwardIterator first, ForwardIterator /*last*/, const LessThanComparable& /*value*/)
{
  system.validate_dispatch();
  return cuda::std::make_pair(first, first);
}

TEST_CASE("ScalarEqualRangeDispatchExplicit", "[binary_search]")
{
  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::equal_range(sys, vec.begin(), vec.end(), 0);

  CHECK(sys.is_valid());
}

template <typename ForwardIterator, typename LessThanComparable>
cuda::std::pair<ForwardIterator, ForwardIterator>
equal_range(my_tag, ForwardIterator first, ForwardIterator /*last*/, const LessThanComparable& /*value*/)
{
  *first = 13;
  return cuda::std::make_pair(first, first);
}

TEST_CASE("ScalarEqualRangeDispatchImplicit", "[binary_search]")
{
  thrust::device_vector<int> vec(1);

  thrust::equal_range(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.end()), 0);

  CHECK(13 == vec.front());
}

_CCCL_DIAG_POP

void TestBoundsWithBigIndexesHelper(int magnitude)
{
  thrust::counting_iterator<long long> begin(1);
  thrust::counting_iterator<long long> end = begin + (1ll << magnitude);
  CHECK(::cuda::std::distance(begin, end) == 1ll << magnitude);

  ::cuda::std::intmax_t distance_low_value =
    ::cuda::std::distance(begin, thrust::lower_bound(thrust::device, begin, end, 17));

  ::cuda::std::intmax_t distance_high_value =
    ::cuda::std::distance(begin, thrust::lower_bound(thrust::device, begin, end, (1ll << magnitude) - 17));

  CHECK(distance_low_value == 16);
  CHECK(distance_high_value == (1ll << magnitude) - 18);

  distance_low_value = ::cuda::std::distance(begin, thrust::upper_bound(thrust::device, begin, end, 17));

  distance_high_value =
    ::cuda::std::distance(begin, thrust::upper_bound(thrust::device, begin, end, (1ll << magnitude) - 17));

  CHECK(distance_low_value == 17);
  CHECK(distance_high_value == (1ll << magnitude) - 17);
}

TEST_CASE("BoundsWithBigIndexes", "[binary_search]")
{
  for (int magnitude : {30, 31, 32, 33})
  {
    TestBoundsWithBigIndexesHelper(magnitude);
  }
}
