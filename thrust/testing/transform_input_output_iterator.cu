#include <thrust/copy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_input_output_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>

#include <unittest/unittest.h>

// There is an unfortunate miscompilation of the gcc-13 vectorizer leading to OOB writes
// Adding this attribute suffices that this miscompilation does not appear anymore
#if _CCCL_COMPILER(GCC, >=, 13)
#  define THRUST_DISABLE_BROKEN_GCC_VECTORIZER __attribute__((optimize("no-tree-vectorize")))
#else // _CCCL_COMPILER(GCC, <, 13)
#  define THRUST_DISABLE_BROKEN_GCC_VECTORIZER
#endif

// ensure that we properly support thrust::transform_input_output_iterator from cuda::std
void TestTransformInputOutputIteratorTraits()
{
  using input_func  = ::cuda::std::negate<int>;
  using output_func = thrust::square<int>;
  using base_it     = thrust::host_vector<int>::iterator;

  using it        = thrust::transform_input_output_iterator<input_func, output_func, base_it>;
  using traits    = cuda::std::iterator_traits<it>;
  using reference = thrust::detail::transform_input_output_iterator_proxy<input_func, output_func, base_it>;

  static_assert(cuda::std::is_same_v<traits::difference_type, ptrdiff_t>);
  static_assert(cuda::std::is_same_v<traits::value_type, int>);
  static_assert(cuda::std::is_same_v<traits::pointer, void>);
  static_assert(cuda::std::is_same_v<traits::reference, reference>);
  static_assert(cuda::std::is_same_v<traits::iterator_category, ::cuda::std::random_access_iterator_tag>);

  static_assert(cuda::std::is_same_v<thrust::iterator_traversal_t<it>, thrust::random_access_traversal_tag>);

  static_assert(cuda::std::__is_cpp17_random_access_iterator<it>::value);

  static_assert(!cuda::std::output_iterator<it, int>);
  static_assert(cuda::std::input_iterator<it>);
  static_assert(cuda::std::forward_iterator<it>);
  static_assert(cuda::std::bidirectional_iterator<it>);
  static_assert(cuda::std::random_access_iterator<it>);
  static_assert(!cuda::std::contiguous_iterator<it>);
}
DECLARE_UNITTEST(TestTransformInputOutputIteratorTraits);

template <class Vector>
THRUST_DISABLE_BROKEN_GCC_VECTORIZER void TestTransformInputOutputIterator()
{
  using T = typename Vector::value_type;

  using InputFunction  = ::cuda::std::negate<T>;
  using OutputFunction = thrust::square<T>;
  using Iterator       = typename Vector::iterator;

  Vector input(4);
  Vector squared(4);
  Vector negated(4);

  // initialize input
  thrust::sequence(input.begin(), input.end(), 1);

  // construct transform_iterator
  thrust::transform_input_output_iterator<InputFunction, OutputFunction, Iterator> transform_iter(
    squared.begin(), InputFunction(), OutputFunction());

  // transform_iter writes squared value
  thrust::copy(input.begin(), input.end(), transform_iter);

  Vector gold_squared{1, 4, 9, 16};

  ASSERT_EQUAL(squared, gold_squared);

  // negated value read from transform_iter
  thrust::copy_n(transform_iter, squared.size(), negated.begin());

  Vector gold_negated{-1, -4, -9, -16};

  ASSERT_EQUAL(negated, gold_negated);
}
DECLARE_VECTOR_UNITTEST(TestTransformInputOutputIterator);

template <class Vector>
THRUST_DISABLE_BROKEN_GCC_VECTORIZER void TestMakeTransformInputOutputIterator()
{
  using T = typename Vector::value_type;

  using InputFunction  = ::cuda::std::negate<T>;
  using OutputFunction = thrust::square<T>;

  Vector input(4);
  Vector negated(4);
  Vector squared(4);

  // initialize input
  thrust::sequence(input.begin(), input.end(), 1);

  // negated value read from transform iterator
  thrust::copy_n(thrust::make_transform_input_output_iterator(input.begin(), InputFunction(), OutputFunction()),
                 input.size(),
                 negated.begin());

  Vector gold_negated{-1, -2, -3, -4};

  ASSERT_EQUAL(negated, gold_negated);

  // squared value written by transform iterator
  thrust::copy(negated.begin(),
               negated.end(),
               thrust::make_transform_input_output_iterator(squared.begin(), InputFunction(), OutputFunction()));

  Vector gold_squared{1, 4, 9, 16};

  ASSERT_EQUAL(squared, gold_squared);
}
DECLARE_VECTOR_UNITTEST(TestMakeTransformInputOutputIterator);

template <typename T>
struct TestTransformInputOutputIteratorScan
{
  void operator()(const size_t n)
  {
    thrust::host_vector<T> h_data   = unittest::random_samples<T>(n);
    thrust::device_vector<T> d_data = h_data;

    thrust::host_vector<T> h_result(n);
    thrust::device_vector<T> d_result(n);

    // run on host (uses forward iterator negate)
    thrust::inclusive_scan(
      thrust::make_transform_input_output_iterator(h_data.begin(), ::cuda::std::negate<T>(), ::cuda::std::identity{}),
      thrust::make_transform_input_output_iterator(h_data.end(), ::cuda::std::negate<T>(), ::cuda::std::identity{}),
      h_result.begin());
    // run on device (uses reverse iterator negate)
    thrust::inclusive_scan(
      d_data.begin(),
      d_data.end(),
      thrust::make_transform_input_output_iterator(d_result.begin(), thrust::square<T>(), ::cuda::std::negate<T>()));

    ASSERT_EQUAL(h_result, d_result);
  }
};
VariableUnitTest<TestTransformInputOutputIteratorScan, IntegralTypes> TestTransformInputOutputIteratorScanInstance;
