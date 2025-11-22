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

TEMPLATE_LIST_TEST_CASE("UnaryNoStencilSimple", "[transform_if]", vector_list)
{
  using Vector = TestType;
  using T      = typename Vector::value_type;

  typename Vector::iterator iter;

  Vector input{0, -2, 0};
  Vector output{-1, -2, -3};
  Vector result{-1, 2, -3};

  iter =
    thrust::transform_if(input.begin(), input.end(), output.begin(), ::cuda::std::negate<T>(), ::cuda::std::identity{});

  CHECK(std::size_t(iter - output.begin()) == input.size());
  CHECK(output == result);
}

template <typename InputIterator, typename ForwardIterator, typename UnaryFunction, typename Predicate>
ForwardIterator
transform_if(my_system& system, InputIterator, InputIterator, ForwardIterator result, UnaryFunction, Predicate)
{
  system.validate_dispatch();
  return result;
}

TEST_CASE("UnaryNoStencilDispatchExplicit", "[transform_if]")
{
  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::transform_if(sys, vec.begin(), vec.begin(), vec.begin(), vec.begin(), 0);

  CHECK(sys.is_valid());
}

template <typename InputIterator, typename ForwardIterator, typename UnaryFunction, typename Predicate>
ForwardIterator transform_if(my_tag, InputIterator, InputIterator, ForwardIterator result, UnaryFunction, Predicate)
{
  *result = 13;
  return result;
}

TEST_CASE("UnaryNoStencilDispatchImplicit", "[transform_if]")
{
  thrust::device_vector<int> vec(1);

  thrust::transform_if(
    thrust::retag<my_tag>(vec.begin()),
    thrust::retag<my_tag>(vec.begin()),
    thrust::retag<my_tag>(vec.begin()),
    thrust::retag<my_tag>(vec.begin()),
    0);

  CHECK(13 == vec.front());
}

TEMPLATE_LIST_TEST_CASE("UnarySimple", "[transform_if]", vector_list)
{
  using Vector = TestType;
  using T      = typename Vector::value_type;

  typename Vector::iterator iter;

  Vector input{1, -2, 3};
  Vector stencil{1, 0, 1};
  Vector output{1, 2, 3};
  Vector result{-1, 2, -3};

  iter = thrust::transform_if(
    input.begin(), input.end(), stencil.begin(), output.begin(), ::cuda::std::negate<T>(), ::cuda::std::identity{});

  CHECK(std::size_t(iter - output.begin()) == input.size());
  CHECK(output == result);
}

template <typename InputIterator1,
          typename InputIterator2,
          typename ForwardIterator,
          typename UnaryFunction,
          typename Predicate>
ForwardIterator
transform_if(my_system& system, InputIterator1, InputIterator1, ForwardIterator result, UnaryFunction, Predicate)
{
  system.validate_dispatch();
  return result;
}

void TestTransformIfUnaryDispatchExplicit()
{
  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::transform_if(sys, vec.begin(), vec.begin(), vec.begin(), 0, 0);

  CHECK(sys.is_valid());
}

template <typename InputIterator1,
          typename InputIterator2,
          typename ForwardIterator,
          typename UnaryFunction,
          typename Predicate>
ForwardIterator transform_if(my_tag, InputIterator1, InputIterator1, ForwardIterator result, UnaryFunction, Predicate)
{
  *result = 13;
  return result;
}

TEST_CASE("UnaryDispatchImplicit", "[transform_if]")
{
  thrust::device_vector<int> vec(1);

  thrust::transform_if(
    thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()), 0, 0);

  CHECK(13 == vec.front());
}

TEMPLATE_LIST_TEST_CASE("BinarySimple", "[transform_if]", vector_list)
{
  using Vector = TestType;
  using T      = typename Vector::value_type;

  typename Vector::iterator iter;

  Vector input1{1, -2, 3};
  Vector input2{-4, 5, 6};
  Vector stencil{0, 1, 0};
  Vector output{1, 2, 3};
  Vector result{5, 2, -3};

  ::cuda::std::identity identity;

  iter = thrust::transform_if(
    input1.begin(),
    input1.end(),
    input2.begin(),
    stencil.begin(),
    output.begin(),
    ::cuda::std::minus<T>(),
    ::cuda::std::not_fn(identity));

  CHECK(std::size_t(iter - output.begin()) == input1.size());
  CHECK(output == result);
}

template <typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename ForwardIterator,
          typename BinaryFunction,
          typename Predicate>
ForwardIterator transform_if(
  my_system& system,
  InputIterator1,
  InputIterator1,
  InputIterator2,
  InputIterator3,
  ForwardIterator result,
  BinaryFunction,
  Predicate)
{
  system.validate_dispatch();
  return result;
}

TEST_CASE("BinaryDispatchExplicit", "[transform_if]")
{
  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::transform_if(sys, vec.begin(), vec.begin(), vec.begin(), vec.begin(), vec.begin(), 0, 0);

  CHECK(sys.is_valid());
}

template <typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename ForwardIterator,
          typename BinaryFunction,
          typename Predicate>
ForwardIterator transform_if(
  my_tag,
  InputIterator1,
  InputIterator1,
  InputIterator2,
  InputIterator3,
  ForwardIterator result,
  BinaryFunction,
  Predicate)
{
  *result = 13;
  return result;
}

TEST_CASE("BinaryDispatchImplicit", "[transform_if]")
{
  thrust::device_vector<int> vec(1);

  thrust::transform_if(
    thrust::retag<my_tag>(vec.begin()),
    thrust::retag<my_tag>(vec.begin()),
    thrust::retag<my_tag>(vec.begin()),
    thrust::retag<my_tag>(vec.begin()),
    thrust::retag<my_tag>(vec.begin()),
    0,
    0);

  CHECK(13 == vec.front());
}

struct is_positive
{
  template <typename T>
  _CCCL_HOST_DEVICE bool operator()(T& x)
  {
    return x > 0;
  }
};

TEMPLATE_LIST_TEST_CASE("UnaryNoStencil", "[transform_if]", variable_list)
{
  using T = TestType;
  for (const size_t n : get_test_sizes())
  {
    thrust::host_vector<T> h_input  = unittest::random_integers<T>(n);
    thrust::host_vector<T> h_output = unittest::random_integers<T>(n);

    thrust::device_vector<T> d_input  = h_input;
    thrust::device_vector<T> d_output = h_output;

    thrust::transform_if(h_input.begin(), h_input.end(), h_output.begin(), ::cuda::std::negate<T>(), is_positive());
    thrust::transform_if(d_input.begin(), d_input.end(), d_output.begin(), ::cuda::std::negate<T>(), is_positive());

    CHECK(h_output == d_output);
  }
}

TEMPLATE_LIST_TEST_CASE("Unary", "[transform_if]", variable_list)
{
  using T = TestType;
  for (const size_t n : get_test_sizes())
  {
    thrust::host_vector<T> h_input   = unittest::random_integers<T>(n);
    thrust::host_vector<T> h_stencil = unittest::random_integers<T>(n);
    thrust::host_vector<T> h_output  = unittest::random_integers<T>(n);

    thrust::device_vector<T> d_input   = h_input;
    thrust::device_vector<T> d_stencil = h_stencil;
    thrust::device_vector<T> d_output  = h_output;

    thrust::transform_if(
      h_input.begin(), h_input.end(), h_stencil.begin(), h_output.begin(), ::cuda::std::negate<T>(), is_positive());

    thrust::transform_if(
      d_input.begin(), d_input.end(), d_stencil.begin(), d_output.begin(), ::cuda::std::negate<T>(), is_positive());

    CHECK(h_output == d_output);
  }
}

TEMPLATE_LIST_TEST_CASE("UnaryToDiscardIterator", "[transform_if]", variable_list)
{
  using T = TestType;
  for (const size_t n : get_test_sizes())
  {
    thrust::host_vector<T> h_input   = unittest::random_integers<T>(n);
    thrust::host_vector<T> h_stencil = unittest::random_integers<T>(n);

    thrust::device_vector<T> d_input   = h_input;
    thrust::device_vector<T> d_stencil = h_stencil;

    thrust::discard_iterator<> h_result = thrust::transform_if(
      h_input.begin(),
      h_input.end(),
      h_stencil.begin(),
      thrust::make_discard_iterator(),
      ::cuda::std::negate<T>(),
      is_positive());

    thrust::discard_iterator<> d_result = thrust::transform_if(
      d_input.begin(),
      d_input.end(),
      d_stencil.begin(),
      thrust::make_discard_iterator(),
      ::cuda::std::negate<T>(),
      is_positive());

    thrust::discard_iterator<> reference(n);

    CHECK((reference == h_result));
    CHECK((reference == d_result));
  }
}

TEMPLATE_LIST_TEST_CASE("Binary", "[transform_if]", variable_list)
{
  using T = TestType;
  for (const size_t n : get_test_sizes())
  {
    thrust::host_vector<T> h_input1  = unittest::random_integers<T>(n);
    thrust::host_vector<T> h_input2  = unittest::random_integers<T>(n);
    thrust::host_vector<T> h_stencil = unittest::random_integers<T>(n);
    thrust::host_vector<T> h_output  = unittest::random_integers<T>(n);

    thrust::device_vector<T> d_input1  = h_input1;
    thrust::device_vector<T> d_input2  = h_input2;
    thrust::device_vector<T> d_stencil = h_stencil;
    thrust::device_vector<T> d_output  = h_output;

    thrust::transform_if(
      h_input1.begin(),
      h_input1.end(),
      h_input2.begin(),
      h_stencil.begin(),
      h_output.begin(),
      ::cuda::std::minus<T>(),
      is_positive());

    thrust::transform_if(
      d_input1.begin(),
      d_input1.end(),
      d_input2.begin(),
      d_stencil.begin(),
      d_output.begin(),
      ::cuda::std::minus<T>(),
      is_positive());

    CHECK(h_output == d_output);

    h_stencil = unittest::random_integers<T>(n);
    d_stencil = h_stencil;

    thrust::transform_if(
      h_input1.begin(),
      h_input1.end(),
      h_input2.begin(),
      h_stencil.begin(),
      h_output.begin(),
      ::cuda::std::multiplies<T>(),
      is_positive());

    thrust::transform_if(
      d_input1.begin(),
      d_input1.end(),
      d_input2.begin(),
      d_stencil.begin(),
      d_output.begin(),
      ::cuda::std::multiplies<T>(),
      is_positive());

    CHECK(h_output == d_output);
  }
}

TEMPLATE_LIST_TEST_CASE("BinaryToDiscardIterator", "[transform_if]", variable_list)
{
  using T = TestType;
  for (const size_t n : get_test_sizes())
  {
    thrust::host_vector<T> h_input1  = unittest::random_integers<T>(n);
    thrust::host_vector<T> h_input2  = unittest::random_integers<T>(n);
    thrust::host_vector<T> h_stencil = unittest::random_integers<T>(n);

    thrust::device_vector<T> d_input1  = h_input1;
    thrust::device_vector<T> d_input2  = h_input2;
    thrust::device_vector<T> d_stencil = h_stencil;

    thrust::discard_iterator<> h_result = thrust::transform_if(
      h_input1.begin(),
      h_input1.end(),
      h_input2.begin(),
      h_stencil.begin(),
      thrust::make_discard_iterator(),
      ::cuda::std::minus<T>(),
      is_positive());

    thrust::discard_iterator<> d_result = thrust::transform_if(
      d_input1.begin(),
      d_input1.end(),
      d_input2.begin(),
      d_stencil.begin(),
      thrust::make_discard_iterator(),
      ::cuda::std::minus<T>(),
      is_positive());

    thrust::discard_iterator<> reference(n);

    CHECK((reference == h_result));
    CHECK((reference == d_result));
  }
}
