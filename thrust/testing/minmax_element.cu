#include <thrust/extrema.h>
#include <thrust/iterator/retag.h>

#include <cuda/iterator>

#include <unittest/unittest.h>

template <class Vector>
void TestMinMaxElementSimple()
{
  Vector data{3, 5, 1, 2, 5, 1};

  REQUIRE(*thrust::minmax_element(data.begin(), data.end()).first == 1);
  REQUIRE(*thrust::minmax_element(data.begin(), data.end()).second == 5);
  REQUIRE(thrust::minmax_element(data.begin(), data.end()).first - data.begin() == 2);
  REQUIRE(thrust::minmax_element(data.begin(), data.end()).second - data.begin() == 1);
}
DECLARE_VECTOR_UNITTEST(TestMinMaxElementSimple);

template <class Vector>
void TestMinMaxElementWithTransform()
{
  using T = typename Vector::value_type;

  Vector data{3, 5, 1, 2, 5, 1};

  REQUIRE(*thrust::minmax_element(thrust::make_transform_iterator(data.begin(), ::cuda::std::negate<T>()),
                                  thrust::make_transform_iterator(data.end(), ::cuda::std::negate<T>()))
             .first
          == -5);
  REQUIRE(*thrust::minmax_element(thrust::make_transform_iterator(data.begin(), ::cuda::std::negate<T>()),
                                  thrust::make_transform_iterator(data.end(), ::cuda::std::negate<T>()))
             .second
          == -1);
}
DECLARE_VECTOR_UNITTEST(TestMinMaxElementWithTransform);

template <typename T>
void TestMinMaxElement(const size_t n)
{
  thrust::host_vector<T> h_data   = unittest::random_samples<T>(n);
  thrust::device_vector<T> d_data = h_data;

  typename thrust::host_vector<T>::iterator h_min;
  typename thrust::host_vector<T>::iterator h_max;
  typename thrust::device_vector<T>::iterator d_min;
  typename thrust::device_vector<T>::iterator d_max;

  h_min = thrust::minmax_element(h_data.begin(), h_data.end()).first;
  d_min = thrust::minmax_element(d_data.begin(), d_data.end()).first;
  h_max = thrust::minmax_element(h_data.begin(), h_data.end()).second;
  d_max = thrust::minmax_element(d_data.begin(), d_data.end()).second;

  REQUIRE(h_min - h_data.begin() == d_min - d_data.begin());
  REQUIRE(h_max - h_data.begin() == d_max - d_data.begin());

  h_max = thrust::minmax_element(h_data.begin(), h_data.end(), ::cuda::std::greater<T>()).first;
  d_max = thrust::minmax_element(d_data.begin(), d_data.end(), ::cuda::std::greater<T>()).first;
  h_min = thrust::minmax_element(h_data.begin(), h_data.end(), ::cuda::std::greater<T>()).second;
  d_min = thrust::minmax_element(d_data.begin(), d_data.end(), ::cuda::std::greater<T>()).second;

  REQUIRE(h_min - h_data.begin() == d_min - d_data.begin());
  REQUIRE(h_max - h_data.begin() == d_max - d_data.begin());
}
DECLARE_VARIABLE_UNITTEST(TestMinMaxElement);

template <typename ForwardIterator>
cuda::std::pair<ForwardIterator, ForwardIterator>
minmax_element(my_system& system, ForwardIterator first, ForwardIterator)
{
  system.validate_dispatch();
  return cuda::std::make_pair(first, first);
}

TEST_CASE("TestMinMaxElementDispatchExplicit", "[minmax_element]")
{
  thrust::device_vector<int> vec(1);

  my_system sys(0); // NOLINT(misc-const-correctness)
  thrust::minmax_element(sys, vec.begin(), vec.end());

  REQUIRE(sys.is_valid());
}

template <typename ForwardIterator>
cuda::std::pair<ForwardIterator, ForwardIterator> minmax_element(my_tag, ForwardIterator first, ForwardIterator)
{
  *first = 13;
  return cuda::std::make_pair(first, first);
}

TEST_CASE("TestMinMaxElementDispatchImplicit", "[minmax_element]")
{
  thrust::device_vector<int> vec(1);

  thrust::minmax_element(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.end()));

  REQUIRE(13 == vec.front());
}

void TestMinMaxElementWithBigIndexesHelper(int magnitude)
{
  using Iter = thrust::counting_iterator<long long>;
  const Iter begin(1);
  const Iter end = begin + (1ll << magnitude);
  REQUIRE(::cuda::std::distance(begin, end) == (1ll << magnitude));

  cuda::std::pair<Iter, Iter> result = thrust::minmax_element(thrust::device, begin, end);
  REQUIRE(*result.first == 1);
  REQUIRE(*result.second == (1ll << magnitude));

  result = thrust::minmax_element(thrust::device, begin, end, ::cuda::std::greater<long long>());
  REQUIRE(*result.second == 1);
  REQUIRE(*result.first == (1ll << magnitude));
}

TEST_CASE("TestMinMaxElementWithBigIndexes", "[minmax_element]")
{
  TestMinMaxElementWithBigIndexesHelper(30);
#ifndef THRUST_FORCE_32_BIT_OFFSET_TYPE
  TestMinMaxElementWithBigIndexesHelper(31);
  TestMinMaxElementWithBigIndexesHelper(32);
  TestMinMaxElementWithBigIndexesHelper(33);
#endif
}

TEST_CASE("TestMinElementCudaIterator", "[minmax_element]")
{
  auto result = thrust::minmax_element(thrust::device, cuda::counting_iterator{0}, cuda::counting_iterator{0} + 100);
  REQUIRE(*result.first == 0);
  REQUIRE(*result.second == 99);
}
