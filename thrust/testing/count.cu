#include <thrust/count.h>
#include <thrust/iterator/retag.h>

#include <unittest/unittest.h>

template <class Vector>
void test_count_simple()
{
  Vector data{1, 1, 0, 0, 1};

  REQUIRE(thrust::count(data.begin(), data.end(), 0) == 2);
  REQUIRE(thrust::count(data.begin(), data.end(), 1) == 3);
  REQUIRE(thrust::count(data.begin(), data.end(), 2) == 0);
}
DECLARE_VECTOR_UNITTEST(test_count_simple);

template <typename T>
void test_count(const size_t n)
{
  thrust::host_vector<T> h_data   = unittest::random_samples<T>(n);
  thrust::device_vector<T> d_data = h_data;

  const size_t cpu_result = thrust::count(h_data.begin(), h_data.end(), T(5));
  const size_t gpu_result = thrust::count(d_data.begin(), d_data.end(), T(5));

  REQUIRE(cpu_result == gpu_result);
}
DECLARE_VARIABLE_UNITTEST(test_count);

template <typename T>
struct greater_than_five
{
  _CCCL_HOST_DEVICE bool operator()(const T& x) const
  {
    return x > 5;
  }
};

template <class Vector>
void test_count_if_simple()
{
  using T = typename Vector::value_type;

  Vector data{1, 6, 1, 9, 2};

  REQUIRE(thrust::count_if(data.begin(), data.end(), greater_than_five<T>()) == 2);
}
DECLARE_VECTOR_UNITTEST(test_count_if_simple);

template <typename T>
void test_count_if(const size_t n)
{
  thrust::host_vector<T> h_data   = unittest::random_samples<T>(n);
  thrust::device_vector<T> d_data = h_data;

  const size_t cpu_result = thrust::count_if(h_data.begin(), h_data.end(), greater_than_five<T>());
  const size_t gpu_result = thrust::count_if(d_data.begin(), d_data.end(), greater_than_five<T>());

  REQUIRE(cpu_result == gpu_result);
}
DECLARE_VARIABLE_UNITTEST(test_count_if);

template <typename Vector>
void test_count_from_const_iterator_simple()
{
  Vector data{1, 1, 0, 0, 1};

  REQUIRE(thrust::count(data.cbegin(), data.cend(), 0) == 2);
  REQUIRE(thrust::count(data.cbegin(), data.cend(), 1) == 3);
  REQUIRE(thrust::count(data.cbegin(), data.cend(), 2) == 0);
}
DECLARE_VECTOR_UNITTEST(test_count_from_const_iterator_simple);

template <typename InputIterator, typename EqualityComparable>
int count(my_system& system, InputIterator, InputIterator, EqualityComparable x)
{
  system.validate_dispatch();
  return x;
}

TEST_CASE("TestCountDispatchExplicit", "[count]")
{
  thrust::device_vector<int> vec(1);

  my_system sys(0); // NOLINT(misc-const-correctness)
  thrust::count(sys, vec.begin(), vec.end(), 13);

  REQUIRE(sys.is_valid());
}

template <typename InputIterator, typename EqualityComparable>
int count(my_tag, InputIterator /*first*/, InputIterator, EqualityComparable x)
{
  return x;
}

TEST_CASE("TestCountDispatchImplicit", "[count]")
{
  thrust::device_vector<int> vec(1);

  auto result = thrust::count(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.end()), 13);

  REQUIRE(13 == result);
}

void test_count_with_big_indexes_helper(int magnitude)
{
  const thrust::counting_iterator<long long> begin(1);
  const thrust::counting_iterator<long long> end = begin + (1ll << magnitude);
  REQUIRE(::cuda::std::distance(begin, end) == (1ll << magnitude));

  const long long result = thrust::count(thrust::device, begin, end, (1ll << magnitude) - 17);

  REQUIRE(result == 1);
}

TEST_CASE("TestCountWithBigIndexes", "[count]")
{
  test_count_with_big_indexes_helper(30);
#ifndef THRUST_FORCE_32_BIT_OFFSET_TYPE
  test_count_with_big_indexes_helper(31);
  test_count_with_big_indexes_helper(32);
  test_count_with_big_indexes_helper(33);
#endif
}
