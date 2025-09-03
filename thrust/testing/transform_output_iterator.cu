#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>

#include <unittest/unittest.h>

// ensure that we properly support thrust::transform_output_iterator from cuda::std
void TestTransformOutputIteratorTraits()
{
  using func    = ::cuda::std::negate<int>;
  using base_it = thrust::host_vector<int>::iterator;

  using it        = thrust::transform_output_iterator<func, base_it>;
  using traits    = cuda::std::iterator_traits<it>;
  using reference = thrust::detail::transform_output_iterator_proxy<func, base_it>;

  static_assert(cuda::std::is_same_v<traits::difference_type, ptrdiff_t>);
  static_assert(cuda::std::is_same_v<traits::value_type, int>);
  static_assert(cuda::std::is_same_v<traits::pointer, void>);
  static_assert(cuda::std::is_same_v<traits::reference, reference>);
  static_assert(cuda::std::is_same_v<traits::iterator_category, ::cuda::std::random_access_iterator_tag>);

  static_assert(cuda::std::is_same_v<thrust::iterator_traversal_t<it>, thrust::random_access_traversal_tag>);

  static_assert(cuda::std::__is_cpp17_random_access_iterator<it>::value);

  static_assert(!cuda::std::output_iterator<it, int>);
  // FIXME(bgruber): all up to and including random access should be true
  static_assert(!cuda::std::input_iterator<it>);
  static_assert(!cuda::std::forward_iterator<it>);
  static_assert(!cuda::std::bidirectional_iterator<it>);
  static_assert(!cuda::std::random_access_iterator<it>);
  static_assert(!cuda::std::contiguous_iterator<it>);
}
DECLARE_UNITTEST(TestTransformOutputIteratorTraits);

template <class Vector>
void TestTransformOutputIterator()
{
  using T = typename Vector::value_type;

  using UnaryFunction = thrust::square<T>;
  using Iterator      = typename Vector::iterator;

  Vector input(4);
  Vector output(4);

  // initialize input
  thrust::sequence(input.begin(), input.end(), T{1});

  // construct transform_iterator
  thrust::transform_output_iterator<UnaryFunction, Iterator> output_iter(output.begin(), UnaryFunction());

  thrust::copy(input.begin(), input.end(), output_iter);

  Vector gold_output{1, 4, 9, 16};

  ASSERT_EQUAL(output, gold_output);
}
DECLARE_VECTOR_UNITTEST(TestTransformOutputIterator);

template <class Vector>
void TestMakeTransformOutputIterator()
{
  using T = typename Vector::value_type;

  using UnaryFunction = thrust::square<T>;

  Vector input(4);
  Vector output(4);

  // initialize input
  thrust::sequence(input.begin(), input.end(), 1);

  thrust::copy(input.begin(), input.end(), thrust::make_transform_output_iterator(output.begin(), UnaryFunction()));

  Vector gold_output{1, 4, 9, 16};
  ASSERT_EQUAL(output, gold_output);
}
DECLARE_VECTOR_UNITTEST(TestMakeTransformOutputIterator);

template <typename T>
struct TestTransformOutputIteratorScan
{
  void operator()(const size_t n)
  {
    thrust::host_vector<T> h_data   = unittest::random_samples<T>(n);
    thrust::device_vector<T> d_data = h_data;

    thrust::host_vector<T> h_result(n);
    thrust::device_vector<T> d_result(n);

    // run on host
    thrust::inclusive_scan(thrust::make_transform_iterator(h_data.begin(), ::cuda::std::negate<T>()),
                           thrust::make_transform_iterator(h_data.end(), ::cuda::std::negate<T>()),
                           h_result.begin());
    // run on device
    thrust::inclusive_scan(
      d_data.begin(), d_data.end(), thrust::make_transform_output_iterator(d_result.begin(), ::cuda::std::negate<T>()));

    ASSERT_EQUAL(h_result, d_result);
  }
};
VariableUnitTest<TestTransformOutputIteratorScan, SignedIntegralTypes> TestTransformOutputIteratorScanInstance;
