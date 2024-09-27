#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/transform_reduce.h>

#include <cuda/std/type_traits>

#include <unittest/unittest.h>

template <class Vector>
void TestPermutationIteratorSimple()
{
  using T        = typename Vector::value_type;
  using Iterator = typename Vector::iterator;

  Vector source(8);
  Vector indices{3, 0, 5, 7};

  // initialize input
  thrust::sequence(source.begin(), source.end(), 1);

  thrust::permutation_iterator<Iterator, Iterator> begin(source.begin(), indices.begin());
  thrust::permutation_iterator<Iterator, Iterator> end(source.begin(), indices.end());

  ASSERT_EQUAL(end - begin, 4);
  ASSERT_EQUAL((begin + 4) == end, true);

  ASSERT_EQUAL((T) *begin, 4);

  begin++;
  end--;

  ASSERT_EQUAL((T) *begin, 1);
  ASSERT_EQUAL((T) *end, 8);
  ASSERT_EQUAL(end - begin, 2);

  end--;

  *begin = 10;
  *end   = 20;

  Vector ref{10, 2, 3, 4, 5, 20, 7, 8};
  ASSERT_EQUAL(source, ref);
}
DECLARE_INTEGRAL_VECTOR_UNITTEST(TestPermutationIteratorSimple);
static_assert(cuda::std::is_trivially_copy_constructible<thrust::permutation_iterator<int*, int*>>::value, "");
static_assert(cuda::std::is_trivially_copyable<thrust::permutation_iterator<int*, int*>>::value, "");

template <class Vector>
void TestPermutationIteratorGather()
{
  using Iterator = typename Vector::iterator;

  Vector source(8);
  Vector indices{3, 0, 5, 7};
  Vector output(4, 10);

  // initialize input
  thrust::sequence(source.begin(), source.end(), 1);

  thrust::permutation_iterator<Iterator, Iterator> p_source(source.begin(), indices.begin());

  thrust::copy(p_source, p_source + 4, output.begin());

  Vector ref{4, 1, 6, 8};
  ASSERT_EQUAL(output, ref);
}
DECLARE_INTEGRAL_VECTOR_UNITTEST(TestPermutationIteratorGather);

template <class Vector>
void TestPermutationIteratorScatter()
{
  using Iterator = typename Vector::iterator;

  Vector source(4, 10);
  Vector indices{3, 0, 5, 7};
  Vector output(8);

  // initialize output
  thrust::sequence(output.begin(), output.end(), 1);

  // construct transform_iterator
  thrust::permutation_iterator<Iterator, Iterator> p_output(output.begin(), indices.begin());

  thrust::copy(source.begin(), source.end(), p_output);

  Vector ref{10, 2, 3, 10, 5, 10, 7, 10};
  ASSERT_EQUAL(output, ref);
}
DECLARE_INTEGRAL_VECTOR_UNITTEST(TestPermutationIteratorScatter);

template <class Vector>
void TestMakePermutationIterator()
{
  Vector source(8);
  Vector indices{3, 0, 5, 7};
  Vector output(4, 10);

  // initialize input
  thrust::sequence(source.begin(), source.end(), 1);

  thrust::copy(thrust::make_permutation_iterator(source.begin(), indices.begin()),
               thrust::make_permutation_iterator(source.begin(), indices.begin()) + 4,
               output.begin());

  Vector ref{4, 1, 6, 8};
  ASSERT_EQUAL(output, ref);
}
DECLARE_INTEGRAL_VECTOR_UNITTEST(TestMakePermutationIterator);

template <typename Vector>
void TestPermutationIteratorReduce()
{
  using T        = typename Vector::value_type;
  using Iterator = typename Vector::iterator;

  Vector source(8);
  Vector indices{3, 0, 5, 7};
  Vector output(4, 10);

  // initialize input
  thrust::sequence(source.begin(), source.end(), 1);

  // construct transform_iterator
  thrust::permutation_iterator<Iterator, Iterator> iter(source.begin(), indices.begin());

  T result1 = thrust::reduce(thrust::make_permutation_iterator(source.begin(), indices.begin()),
                             thrust::make_permutation_iterator(source.begin(), indices.begin()) + 4);

  ASSERT_EQUAL(result1, 19);

  T result2 = thrust::transform_reduce(
    thrust::make_permutation_iterator(source.begin(), indices.begin()),
    thrust::make_permutation_iterator(source.begin(), indices.begin()) + 4,
    thrust::negate<T>(),
    T(0),
    thrust::plus<T>());
  ASSERT_EQUAL(result2, -19);
};
DECLARE_INTEGRAL_VECTOR_UNITTEST(TestPermutationIteratorReduce);

void TestPermutationIteratorHostDeviceGather()
{
  using T              = int;
  using HostVector     = thrust::host_vector<T>;
  using DeviceVector   = thrust::host_vector<T>;
  using HostIterator   = HostVector::iterator;
  using DeviceIterator = DeviceVector::iterator;

  HostVector h_source(8);
  HostVector h_indices{3, 0, 5, 7};
  HostVector h_output(4, 10);

  DeviceVector d_source(8);
  DeviceVector d_indices(h_indices);
  DeviceVector d_output(4, 10);

  // initialize source
  thrust::sequence(h_source.begin(), h_source.end(), 1);
  thrust::sequence(d_source.begin(), d_source.end(), 1);

  thrust::permutation_iterator<HostIterator, HostIterator> p_h_source(h_source.begin(), h_indices.begin());
  thrust::permutation_iterator<DeviceIterator, DeviceIterator> p_d_source(d_source.begin(), d_indices.begin());

  // gather host->device
  thrust::copy(p_h_source, p_h_source + 4, d_output.begin());

  DeviceVector dref{4, 1, 6, 8};
  ASSERT_EQUAL(d_output, dref);

  // gather device->host
  thrust::copy(p_d_source, p_d_source + 4, h_output.begin());

  HostVector href{4, 1, 6, 8};
  ASSERT_EQUAL(h_output, href);
}
DECLARE_UNITTEST(TestPermutationIteratorHostDeviceGather);

void TestPermutationIteratorHostDeviceScatter()
{
  using T              = int;
  using HostVector     = thrust::host_vector<T>;
  using DeviceVector   = thrust::host_vector<T>;
  using HostIterator   = HostVector::iterator;
  using DeviceIterator = DeviceVector::iterator;

  HostVector h_source(4, 10);
  HostVector h_indices{3, 0, 5, 7};
  HostVector h_output(8);

  DeviceVector d_source(4, 10);
  DeviceVector d_indices(h_indices);
  DeviceVector d_output(8);

  // initialize source
  thrust::sequence(h_output.begin(), h_output.end(), 1);
  thrust::sequence(d_output.begin(), d_output.end(), 1);

  thrust::permutation_iterator<HostIterator, HostIterator> p_h_output(h_output.begin(), h_indices.begin());
  thrust::permutation_iterator<DeviceIterator, DeviceIterator> p_d_output(d_output.begin(), d_indices.begin());

  // scatter host->device
  thrust::copy(h_source.begin(), h_source.end(), p_d_output);

  DeviceVector dref{10, 2, 3, 10, 5, 10, 7, 10};
  ASSERT_EQUAL(d_output, dref);

  // scatter device->host
  thrust::copy(d_source.begin(), d_source.end(), p_h_output);

  HostVector href(dref);
  ASSERT_EQUAL(h_output, href);
}
DECLARE_UNITTEST(TestPermutationIteratorHostDeviceScatter);

template <typename Vector>
void TestPermutationIteratorWithCountingIterator()
{
  using T      = typename Vector::value_type;
  using diff_t = typename thrust::counting_iterator<T>::difference_type;

  thrust::counting_iterator<T> input(0), index(0);

  // test copy()
  {
    Vector output(4, 0);

    auto first = thrust::make_permutation_iterator(input, index);
    auto last  = thrust::make_permutation_iterator(input, index + static_cast<diff_t>(output.size()));

    thrust::copy(first, last, output.begin());

    Vector ref{0, 1, 2, 3};
    ASSERT_EQUAL(output, ref);
  }

  // test copy()
  {
    Vector output(4, 0);

    thrust::transform(thrust::make_permutation_iterator(input, index),
                      thrust::make_permutation_iterator(input, index + 4),
                      output.begin(),
                      thrust::identity<T>());

    Vector ref{0, 1, 2, 3};
    ASSERT_EQUAL(output, ref);
  }
}
DECLARE_INTEGRAL_VECTOR_UNITTEST(TestPermutationIteratorWithCountingIterator);
