#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/retag.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>

#include <cuda/std/tuple>
#include <cuda/std/utility>

#include "catch2_test_helper.h"
#include "unittest/random.h"
#include "unittest/special_types.h"

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

    using Tuple1 = cuda::std::tuple<Iterator1, thrust::discard_iterator<>>;
    using Tuple2 = cuda::std::tuple<Iterator2, thrust::discard_iterator<>>;

    using ZipIterator1 = thrust::zip_iterator<Tuple1>;
    using ZipIterator2 = thrust::zip_iterator<Tuple2>;

    ZipIterator1 z1(cuda::std::tuple(h_output.begin(), thrust::make_discard_iterator()));
    ZipIterator2 z2(cuda::std::tuple(d_output.begin(), thrust::make_discard_iterator()));

    ZipIterator1 h_result = thrust::transform(h_input.begin(), h_input.end(), z1, repeat2());

    ZipIterator2 d_result = thrust::transform(d_input.begin(), d_input.end(), z2, repeat2());

    thrust::discard_iterator<> reference(n);

    CHECK(h_output == d_output);

    CHECK(reference == cuda::std::get<1>(h_result.get_iterator_tuple()));
    CHECK(reference == cuda::std::get<1>(d_result.get_iterator_tuple()));
  }
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

TEMPLATE_LIST_TEST_CASE("WithIndirection", "[transform]", integral_vector_list)
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

TEST_CASE("Base", "[transform_n]")
{
  using namespace thrust::placeholders;

  thrust::device_vector vec{1, 2, 3, 4, 5};
  thrust::device_vector stencil{true, false, true};

  {
    thrust::device_vector result{0, 0, 0};
    thrust::transform_n(vec.begin(), 3, result.begin(), _1 * 2);
    CHECK(result == (thrust::device_vector{2, 4, 6}));
  }
  {
    thrust::device_vector result{0, 0, 0};
    thrust::transform_n(vec.begin(), 3, vec.begin(), result.begin(), _1 * _2);
    CHECK(result == (thrust::device_vector{1, 4, 9}));
  }
  {
    thrust::device_vector result{0, 0, 0};
    thrust::transform_if_n(vec.begin(), 3, result.begin(), _1 * 2, _1 > 1);
    CHECK(result == (thrust::device_vector{0, 4, 6}));
  }
  {
    thrust::device_vector result{0, 0, 0};
    thrust::transform_if_n(vec.begin(), 3, stencil.begin(), result.begin(), _1 * 2, _1);
    CHECK(result == (thrust::device_vector{2, 0, 6}));
  }
  {
    thrust::device_vector result{0, 0, 0};
    thrust::transform_if_n(vec.begin(), 3, vec.begin(), stencil.begin(), result.begin(), _1 * _2, _1);
    CHECK(result == (thrust::device_vector{1, 0, 9}));
  }
}
