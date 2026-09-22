#include <thrust/extrema.h>
#include <thrust/iterator/retag.h>

#include <cuda/iterator>

#include <unittest/unittest.h>

template <class Vector>
void TestMinElementSimple()
{
  using T = typename Vector::value_type;

  Vector data{3, 5, 1, 2, 5, 1};

  REQUIRE(*thrust::min_element(data.begin(), data.end()) == 1);
  REQUIRE(thrust::min_element(data.begin(), data.end()) - data.begin() == 2);

  REQUIRE(*thrust::min_element(data.begin(), data.end(), ::cuda::std::greater<T>()) == 5);
  REQUIRE(thrust::min_element(data.begin(), data.end(), ::cuda::std::greater<T>()) - data.begin() == 1);
}
DECLARE_VECTOR_UNITTEST(TestMinElementSimple);

template <class Vector>
void TestMinElementWithTransform()
{
  using T = typename Vector::value_type;

  Vector data{3, 5, 1, 2, 5, 1};

  REQUIRE(*thrust::min_element(thrust::make_transform_iterator(data.begin(), ::cuda::std::negate<T>()),
                               thrust::make_transform_iterator(data.end(), ::cuda::std::negate<T>()))
          == -5);
  REQUIRE(*thrust::min_element(thrust::make_transform_iterator(data.begin(), ::cuda::std::negate<T>()),
                               thrust::make_transform_iterator(data.end(), ::cuda::std::negate<T>()),
                               ::cuda::std::greater<T>())
          == -1);
}
DECLARE_VECTOR_UNITTEST(TestMinElementWithTransform);

template <typename T>
void TestMinElement(const size_t n)
{
  thrust::host_vector<T> h_data   = unittest::random_samples<T>(n);
  thrust::device_vector<T> d_data = h_data;

  const typename thrust::host_vector<T>::iterator h_min   = thrust::min_element(h_data.begin(), h_data.end());
  const typename thrust::device_vector<T>::iterator d_min = thrust::min_element(d_data.begin(), d_data.end());

  REQUIRE(h_min - h_data.begin() == d_min - d_data.begin());

  const typename thrust::host_vector<T>::iterator h_max =
    thrust::min_element(h_data.begin(), h_data.end(), ::cuda::std::greater<T>());
  const typename thrust::device_vector<T>::iterator d_max =
    thrust::min_element(d_data.begin(), d_data.end(), ::cuda::std::greater<T>());

  REQUIRE(h_max - h_data.begin() == d_max - d_data.begin());
}
DECLARE_VARIABLE_UNITTEST(TestMinElement);

template <typename ForwardIterator>
ForwardIterator min_element(my_system& system, ForwardIterator first, ForwardIterator)
{
  system.validate_dispatch();
  return first;
}

TEST_CASE("TestMinElementDispatchExplicit", "[min_element]")
{
  thrust::device_vector<int> vec(1);

  my_system sys(0); // NOLINT(misc-const-correctness)
  thrust::min_element(sys, vec.begin(), vec.end());

  REQUIRE(sys.is_valid());
}

template <typename ForwardIterator>
ForwardIterator min_element(my_tag, ForwardIterator first, ForwardIterator)
{
  *first = 13;
  return first;
}

TEST_CASE("TestMinElementDispatchImplicit", "[min_element]")
{
  thrust::device_vector<int> vec(1);

  thrust::min_element(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.end()));

  REQUIRE(13 == vec.front());
}

void TestMinElementWithBigIndexesHelper(int magnitude)
{
  const thrust::counting_iterator<long long> begin(1);
  const thrust::counting_iterator<long long> end = begin + (1ll << magnitude);
  REQUIRE(::cuda::std::distance(begin, end) == (1ll << magnitude));

  REQUIRE(*thrust::min_element(thrust::device, begin, end, ::cuda::std::greater<long long>()) == (1ll << magnitude));
}

TEST_CASE("TestMinElementWithBigIndexes", "[min_element]")
{
  TestMinElementWithBigIndexesHelper(30);
#ifndef THRUST_FORCE_32_BIT_OFFSET_TYPE
  TestMinElementWithBigIndexesHelper(31);
  TestMinElementWithBigIndexesHelper(32);
  TestMinElementWithBigIndexesHelper(33);
#endif
}

TEST_CASE("TestMinElementCudaIterator", "[min_element]")
{
  auto pos = thrust::min_element(thrust::device, cuda::counting_iterator{0}, cuda::counting_iterator{0} + 100);
  REQUIRE(*pos == 0);
}
