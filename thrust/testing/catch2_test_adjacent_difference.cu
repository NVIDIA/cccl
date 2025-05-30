#include <thrust/adjacent_difference.h>
#include <thrust/device_free.h>
#include <thrust/device_malloc.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/retag.h>

#include "catch2_test_helper.h"
#include <unittest/random.h>
#include <unittest/special_types.h>

TEMPLATE_LIST_TEST_CASE("AdjacentDifferenceSimple", "[adjacent_difference]", vector_list)
{
  using Vector = TestType;
  using T      = typename Vector::value_type;

  Vector input{1, 4, 6, 7};
  Vector output(4);
  typename Vector::iterator result;

  result = thrust::adjacent_difference(input.begin(), input.end(), output.begin());

  CHECK(result - output.begin() == 4);
  Vector ref{1, 3, 2, 1};
  CHECK(output == ref);

  result = thrust::adjacent_difference(input.begin(), input.end(), output.begin(), ::cuda::std::plus<T>());

  CHECK(result - output.begin() == 4);
  ref = {1, 5, 10, 13};
  CHECK(output == ref);

  // test in-place operation, result and first are permitted to be the same
  result = thrust::adjacent_difference(input.begin(), input.end(), input.begin());

  CHECK(result - input.begin() == 4);
  ref = {1, 3, 2, 1};
  CHECK(input == ref);
}

TEMPLATE_LIST_TEST_CASE("AdjacentDifference", "[adjacent_difference]", variable_list)
{
  using T = TestType;
  for (const size_t n : get_test_sizes())
  {
    thrust::host_vector<T> h_input   = unittest::random_samples<T>(n);
    thrust::device_vector<T> d_input = h_input;

    thrust::host_vector<T> h_output(n);
    thrust::device_vector<T> d_output(n);

    typename thrust::host_vector<T>::iterator h_result;
    typename thrust::device_vector<T>::iterator d_result;

    h_result = thrust::adjacent_difference(h_input.begin(), h_input.end(), h_output.begin());
    d_result = thrust::adjacent_difference(d_input.begin(), d_input.end(), d_output.begin());

    CHECK(std::size_t(h_result - h_output.begin()) == n);
    CHECK(std::size_t(d_result - d_output.begin()) == n);
    CHECK(h_output == d_output);

    h_result = thrust::adjacent_difference(h_input.begin(), h_input.end(), h_output.begin(), ::cuda::std::plus<T>());
    d_result = thrust::adjacent_difference(d_input.begin(), d_input.end(), d_output.begin(), ::cuda::std::plus<T>());

    CHECK(std::size_t(h_result - h_output.begin()) == n);
    CHECK(std::size_t(d_result - d_output.begin()) == n);
    CHECK(h_output == d_output);

    // in-place operation
    h_result = thrust::adjacent_difference(h_input.begin(), h_input.end(), h_input.begin(), ::cuda::std::plus<T>());
    d_result = thrust::adjacent_difference(d_input.begin(), d_input.end(), d_input.begin(), ::cuda::std::plus<T>());

    CHECK(std::size_t(h_result - h_input.begin()) == n);
    CHECK(std::size_t(d_result - d_input.begin()) == n);
    CHECK(h_input == h_output); // computed previously
    CHECK(d_input == d_output); // computed previously
  }
}

TEMPLATE_LIST_TEST_CASE("AdjacentDifferenceInPlaceWithRelatedIteratorTypes", "[adjacent_difference]", variable_list)
{
  using T = TestType;
  for (const size_t n : get_test_sizes())
  {
    thrust::host_vector<T> h_input   = unittest::random_samples<T>(n);
    thrust::device_vector<T> d_input = h_input;

    thrust::host_vector<T> h_output(n);
    thrust::device_vector<T> d_output(n);

    typename thrust::host_vector<T>::iterator h_result;
    typename thrust::device_vector<T>::iterator d_result;

    h_result = thrust::adjacent_difference(h_input.begin(), h_input.end(), h_output.begin(), ::cuda::std::plus<T>());
    d_result = thrust::adjacent_difference(d_input.begin(), d_input.end(), d_output.begin(), ::cuda::std::plus<T>());

    // in-place operation with different iterator types
    h_result = thrust::adjacent_difference(h_input.cbegin(), h_input.cend(), h_input.begin(), ::cuda::std::plus<T>());
    d_result = thrust::adjacent_difference(d_input.cbegin(), d_input.cend(), d_input.begin(), ::cuda::std::plus<T>());

    CHECK(std::size_t(h_result - h_input.begin()) == n);
    CHECK(std::size_t(d_result - d_input.begin()) == n);
    CHECK(h_output == h_input); // reference computed previously
    CHECK(d_output == d_input); // reference computed previously
  }
}

TEMPLATE_LIST_TEST_CASE("AdjacentDifferenceDiscardIterator", "[adjacent_difference]", variable_list)
{
  using T = TestType;
  for (const size_t n : get_test_sizes())
  {
    thrust::host_vector<T> h_input   = unittest::random_samples<T>(n);
    thrust::device_vector<T> d_input = h_input;

    thrust::discard_iterator<> h_result =
      thrust::adjacent_difference(h_input.begin(), h_input.end(), thrust::make_discard_iterator());
    thrust::discard_iterator<> d_result =
      thrust::adjacent_difference(d_input.begin(), d_input.end(), thrust::make_discard_iterator());

    thrust::discard_iterator<> reference(n);

    CHECK((reference == h_result));
    CHECK((reference == d_result));
  }
}

template <typename InputIterator, typename OutputIterator>
OutputIterator adjacent_difference(my_system& system, InputIterator, InputIterator, OutputIterator result)
{
  system.validate_dispatch();
  return result;
}

TEST_CASE("AdjacentDifferenceDispatchExplicit", "[adjacent_difference]")
{
  thrust::device_vector<int> d_input(1);

  my_system sys(0);
  thrust::adjacent_difference(sys, d_input.begin(), d_input.end(), d_input.begin());

  CHECK(sys.is_valid());
}

template <typename InputIterator, typename OutputIterator>
OutputIterator adjacent_difference(my_tag, InputIterator, InputIterator, OutputIterator result)
{
  *result = 13;
  return result;
}

TEST_CASE("AdjacentDifferenceDispatchImplicit", "[adjacent_difference]")
{
  thrust::device_vector<int> d_input(1);

  thrust::adjacent_difference(thrust::retag<my_tag>(d_input.begin()),
                              thrust::retag<my_tag>(d_input.end()),
                              thrust::retag<my_tag>(d_input.begin()));

  CHECK(13 == d_input.front());
}
