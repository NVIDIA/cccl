#include <thrust/find_end.h>
#include <thrust/iterator/retag.h>

#include "catch2_test_helper.h"
#include "unittest/special_types.h"

TEMPLATE_LIST_TEST_CASE("FindEndSimple", "[find_end]", vector_list)
{
  using Vector = TestType;
  Vector data{1, 2, 3, 4, 2, 3, 4, 5};

  {
    // Test with a pattern that exists in the data
    Vector pattern{3, 4, 5};
    auto result = thrust::find_end(data.begin(), data.end(), pattern.begin(), pattern.end());
    CHECK(result == data.begin() + 5);
  }

  {
    // Test with a pattern that appears multiple times in the data
    Vector pattern{2, 3, 4};
    auto result = thrust::find_end(data.begin(), data.end(), pattern.begin(), pattern.end());
    CHECK(result == data.begin() + 4);
  }

  {
    // Test with a pattern that does not exist in the data
    Vector pattern{7, 8};
    auto result = thrust::find_end(data.begin(), data.end(), pattern.begin(), pattern.end());
    CHECK(result == data.end());
  }

  {
    // Test with an empty pattern
    Vector pattern{};
    auto result = thrust::find_end(data.begin(), data.end(), pattern.begin(), pattern.end());
    CHECK(result == data.end());
  }

  {
    // Test with a pattern longer than the data
    Vector pattern{1, 2, 3, 4, 2, 3, 4, 5, 6};
    auto result = thrust::find_end(data.begin(), data.end(), pattern.begin(), pattern.end());
    CHECK(result == data.end());
  }

  {
    // Test with a pattern that matches the entire data
    Vector pattern{1, 2, 3, 4, 2, 3, 4, 5};
    auto result = thrust::find_end(data.begin(), data.end(), pattern.begin(), pattern.end());
    CHECK(result == data.begin());
  }
}

template <typename T>
struct negate_equal
{
  _CCCL_HOST_DEVICE bool operator()(const T& a, const T& b) const
  {
    return -a == b;
  }
};

TEMPLATE_LIST_TEST_CASE("FindEndWithCustomPredicate", "[find_end]", integral_vector_list)
{
  using Vector = TestType;
  using T      = typename Vector::value_type;

  Vector data    = {1, 2, 3, 4, 2, 3, 4, 5};
  Vector pattern = {-3, -4, -5};
  auto result    = thrust::find_end(data.begin(), data.end(), pattern.begin(), pattern.end(), negate_equal<T>());
  CHECK(result == data.begin() + 5);
}

template <typename ForwardIterator1, typename ForwardIterator2, typename BinaryPredicate>
ForwardIterator1 find_end(
  my_system& system, ForwardIterator1 first1, ForwardIterator1, ForwardIterator2, ForwardIterator2, BinaryPredicate)
{
  system.validate_dispatch();
  return first1;
}

template <typename ForwardIterator1, typename ForwardIterator2>
ForwardIterator1
find_end(my_system& system, ForwardIterator1 first1, ForwardIterator1, ForwardIterator2, ForwardIterator2)
{
  system.validate_dispatch();
  return first1 + 2;
}

TEST_CASE("FindEndDispatchExplicit", "[find_end]")
{
  thrust::device_vector<int> vec{1, 2, 3, 4};
  thrust::device_vector<int> pattern{2, 3};

  {
    my_system sys(0);
    auto result =
      thrust::find_end(sys, vec.begin(), vec.end(), pattern.begin(), pattern.end(), ::cuda::std::equal_to<int>());
    CHECK(sys.is_valid());
    CHECK(result == vec.begin());
  }

  {
    my_system sys(0);
    auto result = thrust::find_end(sys, vec.begin(), vec.end(), pattern.begin(), pattern.end());
    CHECK(sys.is_valid());
    CHECK(result == vec.begin() + 2);
  }
}

template <typename ForwardIterator1, typename ForwardIterator2, typename BinaryPredicate>
ForwardIterator1
find_end(my_tag, ForwardIterator1 first1, ForwardIterator1, ForwardIterator2, ForwardIterator2, BinaryPredicate)
{
  *first1 = 13;
  return first1;
}

template <typename ForwardIterator1, typename ForwardIterator2>
ForwardIterator1 find_end(my_tag, ForwardIterator1 first1, ForwardIterator1, ForwardIterator2, ForwardIterator2)
{
  *first1 = 42;
  return first1;
}

TEST_CASE("FindEndDispatchImplicit", "[find_end]")
{
  thrust::device_vector<int> vec{1, 2, 3, 4};
  thrust::device_vector<int> pattern{2, 3};

  {
    thrust::find_end(
      thrust::retag<my_tag>(vec.begin()),
      thrust::retag<my_tag>(vec.end()),
      thrust::retag<my_tag>(pattern.begin()),
      thrust::retag<my_tag>(pattern.end()),
      ::cuda::std::equal_to<int>());
    CHECK(vec.front() == 13);
  }

  {
    thrust::find_end(thrust::retag<my_tag>(vec.begin()),
                     thrust::retag<my_tag>(vec.end()),
                     thrust::retag<my_tag>(pattern.begin()),
                     thrust::retag<my_tag>(pattern.end()));
    CHECK(vec.front() == 42);
  }
}
