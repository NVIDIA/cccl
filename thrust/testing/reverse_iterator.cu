#include <thrust/iterator/reverse_iterator.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>

#include <cuda/std/iterator>
#include <cuda/std/type_traits>

#include <unittest/unittest.h>

// ensure that we properly support thrust::reverse_iterator from cuda::std
TEST_CASE("TestReverseIteratorTraits", "[reverse_iterator]")
{
  using base_it = thrust::host_vector<int>::iterator;

  using it       = thrust::reverse_iterator<base_it>;
  using traits   = cuda::std::iterator_traits<it>;
  using category = ::cuda::std::random_access_iterator_tag;

  static_assert(cuda::std::is_same_v<traits::difference_type, ptrdiff_t>);
  static_assert(cuda::std::is_same_v<traits::value_type, int>);
  static_assert(cuda::std::is_same_v<traits::pointer, void>);
  static_assert(cuda::std::is_same_v<traits::reference, int&>);
  static_assert(cuda::std::is_same_v<traits::iterator_category, category>);

  static_assert(cuda::std::is_same_v<thrust::iterator_traversal_t<it>, thrust::random_access_traversal_tag>);

  static_assert(cuda::std::__has_random_access_traversal<it>);

  static_assert(cuda::std::output_iterator<it, int>);
  static_assert(cuda::std::input_iterator<it>);
  static_assert(cuda::std::forward_iterator<it>);
  static_assert(cuda::std::bidirectional_iterator<it>);
  static_assert(cuda::std::random_access_iterator<it>);
  static_assert(!cuda::std::contiguous_iterator<it>);
}

TEST_CASE("TestReverseIteratorCopyConstructor", "[reverse_iterator]")
{
  thrust::host_vector<int> h_v(1, 13);

  const thrust::reverse_iterator<thrust::host_vector<int>::iterator> h_iter0(h_v.end());
  const thrust::reverse_iterator<thrust::host_vector<int>::iterator> h_iter1(h_iter0);

  REQUIRE(h_iter0 == h_iter1);
  REQUIRE(*h_iter0 == *h_iter1);

  thrust::device_vector<int> d_v(1, 13);

  const thrust::reverse_iterator<thrust::device_vector<int>::iterator> d_iter2(d_v.end());
  const thrust::reverse_iterator<thrust::device_vector<int>::iterator> d_iter3(d_iter2);

  REQUIRE(d_iter2 == d_iter3);
  REQUIRE(*d_iter2 == *d_iter3);
}
static_assert(cuda::std::is_trivially_copy_constructible<thrust::reverse_iterator<int*>>::value);
static_assert(cuda::std::is_trivially_copyable<thrust::reverse_iterator<int*>>::value);

TEST_CASE("TestReverseIteratorIncrement", "[reverse_iterator]")
{
  thrust::host_vector<int> h_v(4);
  thrust::sequence(h_v.begin(), h_v.end());

  thrust::reverse_iterator<thrust::host_vector<int>::iterator> h_iter(h_v.end());

  REQUIRE(*h_iter == 3);

  h_iter++;
  REQUIRE(*h_iter == 2);

  h_iter++;
  REQUIRE(*h_iter == 1);

  h_iter++;
  REQUIRE(*h_iter == 0);

  thrust::device_vector<int> d_v(4);
  thrust::sequence(d_v.begin(), d_v.end());

  thrust::reverse_iterator<thrust::device_vector<int>::iterator> d_iter(d_v.end());

  REQUIRE(*d_iter == 3);

  d_iter++;
  REQUIRE(*d_iter == 2);

  d_iter++;
  REQUIRE(*d_iter == 1);

  d_iter++;
  REQUIRE(*d_iter == 0);
}

template <typename Vector>
void test_reverse_iterator_copy()
{
  Vector source{10, 20, 30, 40};

  Vector destination(8, 0); // arm gcc is complaining here

  thrust::copy(
    thrust::make_reverse_iterator(source.end()), thrust::make_reverse_iterator(source.begin()), destination.begin());

  destination.resize(4);
  Vector ref{40, 30, 20, 10};
  REQUIRE(destination == ref);
}
DECLARE_VECTOR_UNITTEST(test_reverse_iterator_copy);

TEST_CASE("TestReverseIteratorExclusiveScanSimple", "[reverse_iterator]")
{
  using T        = int;
  const size_t n = 10;

  thrust::host_vector<T> h_data(n);
  thrust::sequence(h_data.begin(), h_data.end());

  thrust::device_vector<T> d_data = h_data;

  thrust::host_vector<T> h_result(h_data.size());
  thrust::device_vector<T> d_result(d_data.size());

  thrust::exclusive_scan(
    thrust::make_reverse_iterator(h_data.end()), thrust::make_reverse_iterator(h_data.begin()), h_result.begin());

  thrust::exclusive_scan(
    thrust::make_reverse_iterator(d_data.end()), thrust::make_reverse_iterator(d_data.begin()), d_result.begin());

  REQUIRE(h_result == d_result);
}

template <typename T>
struct TestReverseIteratorExclusiveScan
{
  void operator()(const size_t n)
  {
    thrust::host_vector<T> h_data = unittest::random_samples<T>(n);

    thrust::device_vector<T> d_data = h_data;

    thrust::host_vector<T> h_result(n);
    thrust::device_vector<T> d_result(n);

    thrust::exclusive_scan(
      thrust::make_reverse_iterator(h_data.end()), thrust::make_reverse_iterator(h_data.begin()), h_result.begin());

    thrust::exclusive_scan(
      thrust::make_reverse_iterator(d_data.end()), thrust::make_reverse_iterator(d_data.begin()), d_result.begin());

    REQUIRE(h_result == d_result);
  }
};
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES(TestReverseIteratorExclusiveScan, IntegralTypes);
