#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/retag.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <cuda/std/utility>

#include "catch2_test_helper.h"
#include "unittest/random.h"
#include "unittest/special_types.h"

// %PARAM% TEST_ARITY arity 1:2

#if TEST_ARITY == 1

TEMPLATE_LIST_TEST_CASE("UnarySimple", "[transform]", vector_list)
{
  using Vector = TestType;
  using T      = typename Vector::value_type;

  typename Vector::iterator iter;

  Vector input{1, -2, 3};
  Vector output(3);
  Vector result{-1, 2, -3};

  iter = thrust::transform(input.begin(), input.end(), output.begin(), ::cuda::std::negate<T>());

  CHECK(std::size_t(iter - output.begin()) == input.size());
  CHECK(output == result);
}

template <typename InputIterator, typename OutputIterator, typename UnaryFunction>
OutputIterator transform(my_system& system, InputIterator, InputIterator, OutputIterator result, UnaryFunction)
{
  system.validate_dispatch();
  return result;
}

TEST_CASE("UnaryDispatchExplicit", "[transform]", )
{
  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::transform(sys, vec.begin(), vec.begin(), vec.begin(), 0);

  CHECK(sys.is_valid());
}

template <typename InputIterator, typename OutputIterator, typename UnaryFunction>
OutputIterator transform(my_tag, InputIterator, InputIterator, OutputIterator result, UnaryFunction)
{
  *result = 13;
  return result;
}

TEST_CASE("UnaryDispatchImplicit", "[transform]", )
{
  thrust::device_vector<int> vec(1);

  thrust::transform(
    thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()), 0);

  CHECK(13 == vec.front());
}

TEMPLATE_LIST_TEST_CASE("Unary", "[transform]", variable_list)
{
  using T = TestType;
  for (const size_t n : get_test_sizes())
  {
    thrust::host_vector<T> h_input   = unittest::random_integers<T>(n);
    thrust::device_vector<T> d_input = h_input;

    thrust::host_vector<T> h_output(n);
    thrust::device_vector<T> d_output(n);

    thrust::transform(h_input.begin(), h_input.end(), h_output.begin(), ::cuda::std::negate<T>());
    thrust::transform(d_input.begin(), d_input.end(), d_output.begin(), ::cuda::std::negate<T>());

    CHECK(h_output == d_output);
  }
}

TEMPLATE_LIST_TEST_CASE("UnaryToDiscardIterator", "[transform]", variable_list)
{
  using T = TestType;
  for (const size_t n : get_test_sizes())
  {
    thrust::host_vector<T> h_input   = unittest::random_integers<T>(n);
    thrust::device_vector<T> d_input = h_input;

    thrust::discard_iterator<> h_result =
      thrust::transform(h_input.begin(), h_input.end(), thrust::make_discard_iterator(), ::cuda::std::negate<T>());

    thrust::discard_iterator<> d_result =
      thrust::transform(d_input.begin(), d_input.end(), thrust::make_discard_iterator(), ::cuda::std::negate<T>());

    thrust::discard_iterator<> reference(n);

    CHECK((reference == h_result));
    CHECK((reference == d_result));
  }
}

struct repeat2
{
  template <typename T>
  _CCCL_HOST_DEVICE cuda::std::pair<T, T> operator()(T x)
  {
    return cuda::std::make_pair(x, x);
  }
};

TEMPLATE_LIST_TEST_CASE("UnaryToDiscardIteratorZipped", "[transform]", variable_list)
{
  using T = TestType;
  for (const size_t n : get_test_sizes())
  {
    thrust::host_vector<T> h_input   = unittest::random_integers<T>(n);
    thrust::device_vector<T> d_input = h_input;

    thrust::host_vector<T> h_output(n);
    thrust::device_vector<T> d_output(n);

    using Iterator1 = typename thrust::host_vector<T>::iterator;
    using Iterator2 = typename thrust::device_vector<T>::iterator;

    using Tuple1 = thrust::tuple<Iterator1, thrust::discard_iterator<>>;
    using Tuple2 = thrust::tuple<Iterator2, thrust::discard_iterator<>>;

    using ZipIterator1 = thrust::zip_iterator<Tuple1>;
    using ZipIterator2 = thrust::zip_iterator<Tuple2>;

    ZipIterator1 z1(thrust::make_tuple(h_output.begin(), thrust::make_discard_iterator()));
    ZipIterator2 z2(thrust::make_tuple(d_output.begin(), thrust::make_discard_iterator()));

    ZipIterator1 h_result = thrust::transform(h_input.begin(), h_input.end(), z1, repeat2());

    ZipIterator2 d_result = thrust::transform(d_input.begin(), d_input.end(), z2, repeat2());

    thrust::discard_iterator<> reference(n);

    CHECK(h_output == d_output);

    CHECK(reference == thrust::get<1>(h_result.get_iterator_tuple()));
    CHECK(reference == thrust::get<1>(d_result.get_iterator_tuple()));
  }
}

TEMPLATE_LIST_TEST_CASE("UnaryCountingIterator", "[transform]", generic_list)
{
  using T        = TestType;
  size_t const n = 15 * sizeof(T);

  CHECK(T(n) <= unittest::truncate_to_max_representable<T>(n));

  thrust::counting_iterator<T, thrust::host_system_tag> h_first   = thrust::make_counting_iterator<T>(0);
  thrust::counting_iterator<T, thrust::device_system_tag> d_first = thrust::make_counting_iterator<T>(0);

  thrust::host_vector<T> h_result(n);
  thrust::device_vector<T> d_result(n);

  thrust::transform(h_first, h_first + n, h_result.begin(), ::cuda::std::identity{});
  thrust::transform(d_first, d_first + n, d_result.begin(), ::cuda::std::identity{});

  CHECK(h_result == d_result);
}

#else // #TEST_ARITY == 2

TEMPLATE_LIST_TEST_CASE("BinarySimple", "[transform]", vector_list)
{
  using Vector = TestType;
  using T      = typename Vector::value_type;

  typename Vector::iterator iter;

  // There is a strange gcc bug here where it believes we would write out of bounds.
  // It seems to go away if we add one more element that we leave untouched. Luckily 0 - 0 = 0 so all is fine.
  // Note that we still write the element, so it does not hide a functional thrust bug
  Vector input1{1, -2, 3};
  Vector input2{-4, 5, 6};
  Vector output(3);
  Vector result{5, -7, -3};

  iter = thrust::transform(input1.begin(), input1.end(), input2.begin(), output.begin(), ::cuda::std::minus<T>());

  CHECK(std::size_t(iter - output.begin()) == input1.size());
  CHECK(output == result);
}

template <typename InputIterator1, typename InputIterator2, typename OutputIterator, typename UnaryFunction>
OutputIterator
transform(my_system& system, InputIterator1, InputIterator1, InputIterator2, OutputIterator result, UnaryFunction)
{
  system.validate_dispatch();
  return result;
}

TEST_CASE("BinaryDispatchExplicit", "[transform]")
{
  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::transform(sys, vec.begin(), vec.begin(), vec.begin(), vec.begin(), 0);

  CHECK(sys.is_valid());
}

template <typename InputIterator1, typename InputIterator2, typename OutputIterator, typename UnaryFunction>
OutputIterator transform(my_tag, InputIterator1, InputIterator1, InputIterator2, OutputIterator result, UnaryFunction)
{
  *result = 13;
  return result;
}

TEST_CASE("BinaryDispatchImplicit", "[transform]")
{
  thrust::device_vector<int> vec(1);

  thrust::transform(
    thrust::retag<my_tag>(vec.begin()),
    thrust::retag<my_tag>(vec.begin()),
    thrust::retag<my_tag>(vec.begin()),
    thrust::retag<my_tag>(vec.begin()),
    0);

  CHECK(13 == vec.front());
}

TEMPLATE_LIST_TEST_CASE("Binary", "[transform]", variable_list)
{
  using T = TestType;
  for (const size_t n : get_test_sizes())
  {
    thrust::host_vector<T> h_input1   = unittest::random_integers<T>(n);
    thrust::host_vector<T> h_input2   = unittest::random_integers<T>(n);
    thrust::device_vector<T> d_input1 = h_input1;
    thrust::device_vector<T> d_input2 = h_input2;

    thrust::host_vector<T> h_output(n);
    thrust::device_vector<T> d_output(n);

    thrust::transform(h_input1.begin(), h_input1.end(), h_input2.begin(), h_output.begin(), ::cuda::std::minus<T>());
    thrust::transform(d_input1.begin(), d_input1.end(), d_input2.begin(), d_output.begin(), ::cuda::std::minus<T>());

    CHECK(h_output == d_output);

    thrust::transform(
      h_input1.begin(), h_input1.end(), h_input2.begin(), h_output.begin(), ::cuda::std::multiplies<T>());
    thrust::transform(
      d_input1.begin(), d_input1.end(), d_input2.begin(), d_output.begin(), ::cuda::std::multiplies<T>());

    CHECK(h_output == d_output);
  }
}

TEMPLATE_LIST_TEST_CASE("BinaryToDiscardIterator", "[transform]", variable_list)
{
  using T = TestType;
  for (const size_t n : get_test_sizes())
  {
    thrust::host_vector<T> h_input1   = unittest::random_integers<T>(n);
    thrust::host_vector<T> h_input2   = unittest::random_integers<T>(n);
    thrust::device_vector<T> d_input1 = h_input1;
    thrust::device_vector<T> d_input2 = h_input2;

    thrust::discard_iterator<> h_result = thrust::transform(
      h_input1.begin(), h_input1.end(), h_input2.begin(), thrust::make_discard_iterator(), ::cuda::std::minus<T>());
    thrust::discard_iterator<> d_result = thrust::transform(
      d_input1.begin(), d_input1.end(), d_input2.begin(), thrust::make_discard_iterator(), ::cuda::std::minus<T>());

    thrust::discard_iterator<> reference(n);

    CHECK((reference == h_result));
    CHECK((reference == d_result));
  }
}

TEMPLATE_LIST_TEST_CASE("BinaryCountingIterator", "[transform]", generic_list)
{
  using T        = TestType;
  size_t const n = 15 * sizeof(T);

  CHECK(T(n) <= unittest::truncate_to_max_representable<T>(n));

  thrust::counting_iterator<T, thrust::host_system_tag> h_first   = thrust::make_counting_iterator<T>(0);
  thrust::counting_iterator<T, thrust::device_system_tag> d_first = thrust::make_counting_iterator<T>(0);

  thrust::host_vector<T> h_result(n);
  thrust::device_vector<T> d_result(n);

  thrust::transform(h_first, h_first + n, h_first, h_result.begin(), ::cuda::std::plus<T>());
  thrust::transform(d_first, d_first + n, d_first, d_result.begin(), ::cuda::std::plus<T>());

  CHECK(h_result == d_result);
}

template <typename T>
struct plus_mod3
{
  T* table;

  _CCCL_HOST_DEVICE T operator()(T a, T b)
  {
    return table[(int) (a + b)];
  }
};

TEMPLATE_LIST_TEST_CASE("BinaryWithIndirection", "[transform]", integral_vector_list)
{
  // add numbers modulo 3 with external lookup table
  using Vector = TestType;
  using T      = typename Vector::value_type;

  Vector input1{0, 1, 2, 1, 2, 0, 1};
  Vector input2{2, 2, 2, 0, 2, 1, 0};
  Vector output(7, 0);

  Vector table{0, 1, 2, 0, 1, 2};

  thrust::transform(
    input1.begin(), input1.end(), input2.begin(), output.begin(), plus_mod3<T>{thrust::raw_pointer_cast(&table[0])});

  Vector ref{2, 0, 1, 1, 1, 1, 1};
  CHECK(output == ref);
}

#endif // #TEST_ARITY == 2
