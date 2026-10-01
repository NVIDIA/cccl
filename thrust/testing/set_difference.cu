#include <thrust/extrema.h>
#include <thrust/functional.h>
#include <thrust/iterator/retag.h>
#include <thrust/set_operations.h>
#include <thrust/sort.h>

#include <unittest/unittest.h>

template <typename InputIterator1, typename InputIterator2, typename OutputIterator>
OutputIterator
set_difference(my_system& system, InputIterator1, InputIterator1, InputIterator2, InputIterator2, OutputIterator result)
{
  system.validate_dispatch();
  return result;
}

TEST_CASE("TestSetDifferenceDispatchExplicit", "[set_difference]")
{
  thrust::device_vector<int> vec(1);

  my_system sys(0); // NOLINT(misc-const-correctness)
  thrust::set_difference(sys, vec.begin(), vec.begin(), vec.begin(), vec.begin(), vec.begin());

  REQUIRE(sys.is_valid());
}

template <typename InputIterator1, typename InputIterator2, typename OutputIterator>
OutputIterator
set_difference(my_tag, InputIterator1, InputIterator1, InputIterator2, InputIterator2, OutputIterator result)
{
  *result = 13;
  return result;
}

TEST_CASE("TestSetDifferenceDispatchImplicit", "[set_difference]")
{
  thrust::device_vector<int> vec(1);

  thrust::set_difference(
    thrust::retag<my_tag>(vec.begin()),
    thrust::retag<my_tag>(vec.begin()),
    thrust::retag<my_tag>(vec.begin()),
    thrust::retag<my_tag>(vec.begin()),
    thrust::retag<my_tag>(vec.begin()));

  REQUIRE(13 == vec.front());
}

template <typename Vector>
void test_set_difference_simple()
{
  using Iterator = typename Vector::iterator;

  Vector a{0, 2, 4, 5}, b{0, 3, 3, 4, 6};
  Vector ref{2, 5};
  Vector result(2);

  const Iterator end = thrust::set_difference(a.begin(), a.end(), b.begin(), b.end(), result.begin());

  REQUIRE(result.end() == end);
  REQUIRE(ref == result);
}
DECLARE_VECTOR_UNITTEST(test_set_difference_simple);

template <typename T>
void test_set_difference(const size_t n)
{
  size_t sizes[]         = {0, 1, n / 2, n, n + 1, 2 * n};
  const size_t num_sizes = sizeof(sizes) / sizeof(size_t);

  thrust::host_vector<T> random =
    unittest::random_integers<unittest::int8_t>(n + *thrust::max_element(sizes, sizes + num_sizes));

  thrust::host_vector<T> h_a(random.begin(), random.begin() + n);
  thrust::host_vector<T> h_b(random.begin() + n, random.end());

  thrust::stable_sort(h_a.begin(), h_a.end());
  thrust::stable_sort(h_b.begin(), h_b.end());

  thrust::device_vector<T> d_a = h_a;
  thrust::device_vector<T> d_b = h_b;

  for (const size_t size : sizes)
  {
    thrust::host_vector<T> h_result(n + size);
    thrust::device_vector<T> d_result(n + size);

    typename thrust::host_vector<T>::iterator h_end;
    typename thrust::device_vector<T>::iterator d_end;

    h_end = thrust::set_difference(h_a.begin(), h_a.end(), h_b.begin(), h_b.begin() + size, h_result.begin());
    h_result.resize(h_end - h_result.begin());

    d_end = thrust::set_difference(d_a.begin(), d_a.end(), d_b.begin(), d_b.begin() + size, d_result.begin());
    d_result.resize(d_end - d_result.begin());

    REQUIRE(h_result == d_result);
  }
}
DECLARE_VARIABLE_UNITTEST(test_set_difference);

template <typename T>
void test_set_difference_equivalent_ranges(const size_t n)
{
  const thrust::host_vector<T> temp = unittest::random_integers<T>(n);
  thrust::host_vector<T> h_a        = temp;
  thrust::sort(h_a.begin(), h_a.end());
  thrust::host_vector<T> h_b = h_a;

  thrust::device_vector<T> d_a = h_a;
  thrust::device_vector<T> d_b = h_b;

  thrust::host_vector<T> h_result(n);
  thrust::device_vector<T> d_result(n);

  typename thrust::host_vector<T>::iterator h_end;
  typename thrust::device_vector<T>::iterator d_end;

  h_end = thrust::set_difference(h_a.begin(), h_a.end(), h_b.begin(), h_b.end(), h_result.begin());
  h_result.resize(h_end - h_result.begin());

  d_end = thrust::set_difference(d_a.begin(), d_a.end(), d_b.begin(), d_b.end(), d_result.begin());

  d_result.resize(d_end - d_result.begin());

  REQUIRE(h_result == d_result);
}
DECLARE_VARIABLE_UNITTEST(test_set_difference_equivalent_ranges);

template <typename T>
void test_set_difference_multiset(const size_t n)
{
  thrust::host_vector<T> vec = unittest::random_integers<int>(2 * n);

  // restrict elements to [min,13)
  for (typename thrust::host_vector<T>::iterator i = vec.begin(); i != vec.end(); ++i)
  {
    int temp = static_cast<int>(*i);
    temp %= 13;
    *i = temp;
  }

  thrust::host_vector<T> h_a(vec.begin(), vec.begin() + n);
  thrust::host_vector<T> h_b(vec.begin() + n, vec.end());

  thrust::sort(h_a.begin(), h_a.end());
  thrust::sort(h_b.begin(), h_b.end());

  thrust::device_vector<T> d_a = h_a;
  thrust::device_vector<T> d_b = h_b;

  thrust::host_vector<T> h_result(n);
  thrust::device_vector<T> d_result(n);

  typename thrust::host_vector<T>::iterator h_end;
  typename thrust::device_vector<T>::iterator d_end;

  h_end = thrust::set_difference(h_a.begin(), h_a.end(), h_b.begin(), h_b.end(), h_result.begin());
  h_result.resize(h_end - h_result.begin());

  d_end = thrust::set_difference(d_a.begin(), d_a.end(), d_b.begin(), d_b.end(), d_result.begin());

  d_result.resize(d_end - d_result.begin());

  REQUIRE(h_result == d_result);
}
DECLARE_VARIABLE_UNITTEST(test_set_difference_multiset);

// FIXME: disabled on Windows, because it causes a failure on the internal CI system in one specific configuration.
// That failure will be tracked in a new NVBug, this is disabled to unblock submitting all the other changes.
#if !_CCCL_COMPILER(MSVC)
void test_set_difference_with_big_indexes_helper(int magnitude)
{
  const thrust::counting_iterator<long long> begin(0);
  const thrust::counting_iterator<long long> end        = begin + (1ll << magnitude);
  const thrust::counting_iterator<long long> end_longer = end + 1;
  REQUIRE(::cuda::std::distance(begin, end) == (1ll << magnitude));

  thrust::device_vector<long long> result;
  result.resize(1);
  thrust::set_difference(thrust::device, begin, end_longer, begin, end, result.begin());

  thrust::host_vector<long long> expected{*end};

  REQUIRE(result == expected);
}

TEST_CASE("TestSetDifferenceWithBigIndexes", "[set_difference]")
{
#  ifndef THRUST_FORCE_32_BIT_OFFSET_TYPE
  test_set_difference_with_big_indexes_helper(30);
  test_set_difference_with_big_indexes_helper(31);
  test_set_difference_with_big_indexes_helper(32);
  test_set_difference_with_big_indexes_helper(33);
#  endif
}

#endif
