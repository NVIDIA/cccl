#include <thrust/binary_search.h>
#include <thrust/detail/allocator/allocator_traits.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <unittest/unittest.h>

//////////////////////
// Vector Functions //
//////////////////////

// convert xxx_vector<T1> to xxx_vector<T2>
template <class ExampleVector, typename NewType>
struct vector_like
{
  using alloc        = typename ExampleVector::allocator_type;
  using alloc_traits = typename thrust::detail::allocator_traits<alloc>;
  using new_alloc    = typename alloc_traits::template rebind_alloc<NewType>;
  using type         = thrust::detail::vector_base<NewType, new_alloc>;
};

template <class Vector>
void TestVectorLowerBoundDescendingSimple()
{
  using T = typename Vector::value_type;

  Vector vec(5);

  vec[0] = 8;
  vec[1] = 7;
  vec[2] = 5;
  vec[3] = 2;
  vec[4] = 0;

  Vector input(10);
  thrust::sequence(input.begin(), input.end());

  using int_type  = typename Vector::difference_type;
  using IntVector = typename vector_like<Vector, int_type>::type;

  // test with integral output type
  IntVector integral_output(10);
  typename IntVector::iterator output_end = thrust::lower_bound(
    vec.begin(), vec.end(), input.begin(), input.end(), integral_output.begin(), thrust::greater<T>());

  ASSERT_EQUAL_QUIET(integral_output.end(), output_end);

  ASSERT_EQUAL(4, integral_output[0]);
  ASSERT_EQUAL(4, integral_output[1]);
  ASSERT_EQUAL(3, integral_output[2]);
  ASSERT_EQUAL(3, integral_output[3]);
  ASSERT_EQUAL(3, integral_output[4]);
  ASSERT_EQUAL(2, integral_output[5]);
  ASSERT_EQUAL(2, integral_output[6]);
  ASSERT_EQUAL(1, integral_output[7]);
  ASSERT_EQUAL(0, integral_output[8]);
  ASSERT_EQUAL(0, integral_output[9]);
}
DECLARE_VECTOR_UNITTEST(TestVectorLowerBoundDescendingSimple);

template <class Vector>
void TestVectorUpperBoundDescendingSimple()
{
  Vector vec(5);

  vec[0] = 8;
  vec[1] = 7;
  vec[2] = 5;
  vec[3] = 2;
  vec[4] = 0;

  Vector input(10);
  thrust::sequence(input.begin(), input.end());

  using int_type  = typename Vector::difference_type;
  using T         = typename Vector::value_type;
  using IntVector = typename vector_like<Vector, int_type>::type;

  // test with integral output type
  IntVector integral_output(10);
  typename IntVector::iterator output_end = thrust::upper_bound(
    vec.begin(), vec.end(), input.begin(), input.end(), integral_output.begin(), thrust::greater<T>());

  ASSERT_EQUAL_QUIET(output_end, integral_output.end());

  ASSERT_EQUAL(5, integral_output[0]);
  ASSERT_EQUAL(4, integral_output[1]);
  ASSERT_EQUAL(4, integral_output[2]);
  ASSERT_EQUAL(3, integral_output[3]);
  ASSERT_EQUAL(3, integral_output[4]);
  ASSERT_EQUAL(3, integral_output[5]);
  ASSERT_EQUAL(2, integral_output[6]);
  ASSERT_EQUAL(2, integral_output[7]);
  ASSERT_EQUAL(1, integral_output[8]);
  ASSERT_EQUAL(0, integral_output[9]);
}
DECLARE_VECTOR_UNITTEST(TestVectorUpperBoundDescendingSimple);

template <class Vector>
void TestVectorBinarySearchDescendingSimple()
{
  Vector vec(5);

  vec[0] = 8;
  vec[1] = 7;
  vec[2] = 5;
  vec[3] = 2;
  vec[4] = 0;

  Vector input(10);
  thrust::sequence(input.begin(), input.end());

  using BoolVector = typename vector_like<Vector, bool>::type;
  using int_type   = typename Vector::difference_type;
  using T          = typename Vector::value_type;
  using IntVector  = typename vector_like<Vector, int_type>::type;

  // test with boolean output type
  BoolVector bool_output(10);
  typename BoolVector::iterator bool_output_end = thrust::binary_search(
    vec.begin(), vec.end(), input.begin(), input.end(), bool_output.begin(), thrust::greater<T>());

  ASSERT_EQUAL_QUIET(bool_output_end, bool_output.end());

  ASSERT_EQUAL(true, bool_output[0]);
  ASSERT_EQUAL(false, bool_output[1]);
  ASSERT_EQUAL(true, bool_output[2]);
  ASSERT_EQUAL(false, bool_output[3]);
  ASSERT_EQUAL(false, bool_output[4]);
  ASSERT_EQUAL(true, bool_output[5]);
  ASSERT_EQUAL(false, bool_output[6]);
  ASSERT_EQUAL(true, bool_output[7]);
  ASSERT_EQUAL(true, bool_output[8]);
  ASSERT_EQUAL(false, bool_output[9]);

  // test with integral output type
  IntVector integral_output(10, 2);
  typename IntVector::iterator int_output_end = thrust::binary_search(
    vec.begin(), vec.end(), input.begin(), input.end(), integral_output.begin(), thrust::greater<T>());

  ASSERT_EQUAL_QUIET(int_output_end, integral_output.end());

  ASSERT_EQUAL(1, integral_output[0]);
  ASSERT_EQUAL(0, integral_output[1]);
  ASSERT_EQUAL(1, integral_output[2]);
  ASSERT_EQUAL(0, integral_output[3]);
  ASSERT_EQUAL(0, integral_output[4]);
  ASSERT_EQUAL(1, integral_output[5]);
  ASSERT_EQUAL(0, integral_output[6]);
  ASSERT_EQUAL(1, integral_output[7]);
  ASSERT_EQUAL(1, integral_output[8]);
  ASSERT_EQUAL(0, integral_output[9]);
}
DECLARE_VECTOR_UNITTEST(TestVectorBinarySearchDescendingSimple);

template <typename T>
struct TestVectorLowerBoundDescending
{
  void operator()(const size_t n)
  {
    thrust::host_vector<T> h_vec = unittest::random_integers<T>(n);
    thrust::sort(h_vec.begin(), h_vec.end(), thrust::greater<T>());
    thrust::device_vector<T> d_vec = h_vec;

    thrust::host_vector<T> h_input   = unittest::random_integers<T>(2 * n);
    thrust::device_vector<T> d_input = h_input;

    using int_type = typename thrust::host_vector<T>::difference_type;
    thrust::host_vector<int_type> h_output(2 * n);
    thrust::device_vector<int_type> d_output(2 * n);

    thrust::lower_bound(
      h_vec.begin(), h_vec.end(), h_input.begin(), h_input.end(), h_output.begin(), thrust::greater<T>());
    thrust::lower_bound(
      d_vec.begin(), d_vec.end(), d_input.begin(), d_input.end(), d_output.begin(), thrust::greater<T>());

    ASSERT_EQUAL(h_output, d_output);
  }
};
VariableUnitTest<TestVectorLowerBoundDescending, SignedIntegralTypes> TestVectorLowerBoundDescendingInstance;

template <typename T>
struct TestVectorUpperBoundDescending
{
  void operator()(const size_t n)
  {
    thrust::host_vector<T> h_vec = unittest::random_integers<T>(n);
    thrust::sort(h_vec.begin(), h_vec.end(), thrust::greater<T>());
    thrust::device_vector<T> d_vec = h_vec;

    thrust::host_vector<T> h_input   = unittest::random_integers<T>(2 * n);
    thrust::device_vector<T> d_input = h_input;

    using int_type = typename thrust::host_vector<T>::difference_type;
    thrust::host_vector<int_type> h_output(2 * n);
    thrust::device_vector<int_type> d_output(2 * n);

    thrust::upper_bound(
      h_vec.begin(), h_vec.end(), h_input.begin(), h_input.end(), h_output.begin(), thrust::greater<T>());
    thrust::upper_bound(
      d_vec.begin(), d_vec.end(), d_input.begin(), d_input.end(), d_output.begin(), thrust::greater<T>());

    ASSERT_EQUAL(h_output, d_output);
  }
};
VariableUnitTest<TestVectorUpperBoundDescending, SignedIntegralTypes> TestVectorUpperBoundDescendingInstance;

template <typename T>
struct TestVectorBinarySearchDescending
{
  void operator()(const size_t n)
  {
    thrust::host_vector<T> h_vec = unittest::random_integers<T>(n);
    thrust::sort(h_vec.begin(), h_vec.end(), thrust::greater<T>());
    thrust::device_vector<T> d_vec = h_vec;

    thrust::host_vector<T> h_input   = unittest::random_integers<T>(2 * n);
    thrust::device_vector<T> d_input = h_input;

    using int_type = typename thrust::host_vector<T>::difference_type;
    thrust::host_vector<int_type> h_output(2 * n);
    thrust::device_vector<int_type> d_output(2 * n);

    thrust::binary_search(
      h_vec.begin(), h_vec.end(), h_input.begin(), h_input.end(), h_output.begin(), thrust::greater<T>());
    thrust::binary_search(
      d_vec.begin(), d_vec.end(), d_input.begin(), d_input.end(), d_output.begin(), thrust::greater<T>());

    ASSERT_EQUAL(h_output, d_output);
  }
};
VariableUnitTest<TestVectorBinarySearchDescending, SignedIntegralTypes> TestVectorBinarySearchDescendingInstance;
