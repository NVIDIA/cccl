#include <thrust/extrema.h>
#include <thrust/functional.h>
#include <thrust/iterator/retag.h>
#include <thrust/iterator/transform_iterator.h>

#include <cuda/iterator>

#include <unittest/unittest.h>

template <class Vector>
void test_max_element_simple()
{
  using T = typename Vector::value_type;

  Vector data{3, 5, 1, 2, 5, 1};

  REQUIRE(*thrust::max_element(data.begin(), data.end()) == 5);
  REQUIRE(thrust::max_element(data.begin(), data.end()) - data.begin() == 1);

  REQUIRE(*thrust::max_element(data.begin(), data.end(), ::cuda::std::greater<T>()) == 1);
  REQUIRE(thrust::max_element(data.begin(), data.end(), ::cuda::std::greater<T>()) - data.begin() == 2);
}
DECLARE_VECTOR_UNITTEST(test_max_element_simple);

template <class Vector>
void test_max_element_with_transform()
{
  using T = typename Vector::value_type;

  Vector data{3, 5, 1, 2, 5, 1};

  REQUIRE(*thrust::max_element(thrust::make_transform_iterator(data.begin(), ::cuda::std::negate<T>()),
                               thrust::make_transform_iterator(data.end(), ::cuda::std::negate<T>()))
          == -1);
  REQUIRE(*thrust::max_element(thrust::make_transform_iterator(data.begin(), ::cuda::std::negate<T>()),
                               thrust::make_transform_iterator(data.end(), ::cuda::std::negate<T>()),
                               ::cuda::std::greater<T>())
          == -5);
}
DECLARE_VECTOR_UNITTEST(test_max_element_with_transform);

template <typename T>
void test_max_element(const size_t n)
{
  thrust::host_vector<T> h_data   = unittest::random_samples<T>(n);
  thrust::device_vector<T> d_data = h_data;

  const typename thrust::host_vector<T>::iterator h_max   = thrust::max_element(h_data.begin(), h_data.end());
  const typename thrust::device_vector<T>::iterator d_max = thrust::max_element(d_data.begin(), d_data.end());

  REQUIRE(h_max - h_data.begin() == d_max - d_data.begin());

  const typename thrust::host_vector<T>::iterator h_min =
    thrust::max_element(h_data.begin(), h_data.end(), ::cuda::std::greater<T>());
  const typename thrust::device_vector<T>::iterator d_min =
    thrust::max_element(d_data.begin(), d_data.end(), ::cuda::std::greater<T>());

  REQUIRE(h_min - h_data.begin() == d_min - d_data.begin());
}
DECLARE_VARIABLE_UNITTEST(test_max_element);

template <typename ForwardIterator>
ForwardIterator max_element(my_system& system, ForwardIterator first, ForwardIterator)
{
  system.validate_dispatch();
  return first;
}

TEST_CASE("TestMaxElementDispatchExplicit", "[max_element]")
{
  thrust::device_vector<int> vec(1);

  my_system sys(0); // NOLINT(misc-const-correctness)
  thrust::max_element(sys, vec.begin(), vec.end());

  REQUIRE(sys.is_valid());
}

template <typename ForwardIterator>
ForwardIterator max_element(my_tag, ForwardIterator first, ForwardIterator)
{
  *first = 13;
  return first;
}

TEST_CASE("TestMaxElementDispatchImplicit", "[max_element]")
{
  thrust::device_vector<int> vec(1);

  thrust::max_element(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.end()));

  REQUIRE(13 == vec.front());
}

void test_max_element_with_big_indexes_helper(int magnitude)
{
  const thrust::counting_iterator<long long> begin(1);
  const thrust::counting_iterator<long long> end = begin + (1ll << magnitude);
  REQUIRE(::cuda::std::distance(begin, end) == (1ll << magnitude));

  REQUIRE(*thrust::max_element(thrust::device, begin, end) == (1ll << magnitude));
}

TEST_CASE("TestMaxElementWithBigIndexes", "[max_element]")
{
  test_max_element_with_big_indexes_helper(30);
#ifndef THRUST_FORCE_32_BIT_OFFSET_TYPE
  test_max_element_with_big_indexes_helper(31);
  test_max_element_with_big_indexes_helper(32);
  test_max_element_with_big_indexes_helper(33);
#endif
}

TEST_CASE("TestMaxElementCudaIterator", "[max_element]")
{
  auto pos = thrust::max_element(thrust::device, cuda::counting_iterator{0}, cuda::counting_iterator{0} + 100);
  REQUIRE(*pos == 99);
}
