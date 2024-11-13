#include <thrust/binary_search.h>
#include <thrust/distance.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sort.h>

#include <cuda/std/iterator>
#include <cuda/std/type_traits>

#include <cstdint>

#include <unittest/unittest.h>

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_MSVC(4244 4267) // possible loss of data

// ensure that we properly support thrust::counting_iterator from cuda::std
void test_iterator_traits()
{
  using It       = cuda::std::iterator_traits<thrust::counting_iterator<int>>;
  using category = thrust::detail::iterator_category_with_system_and_traversal<std::random_access_iterator_tag,
                                                                               thrust::any_system_tag,
                                                                               thrust::random_access_traversal_tag>;

  static_assert(cuda::std::is_same<It::difference_type, ptrdiff_t>::value, "");
  static_assert(cuda::std::is_same<It::value_type, int>::value, "");
  static_assert(cuda::std::is_same<It::pointer, void>::value, "");
  static_assert(cuda::std::is_same<It::reference, signed int>::value, "");
  static_assert(cuda::std::is_same<It::iterator_category, category>::value, "");

  static_assert(cuda::std::__is_cpp17_random_access_iterator<thrust::counting_iterator<int>>::value, "");
}

template <typename T>
void TestCountingDefaultConstructor()
{
  thrust::counting_iterator<T> iter0;
  ASSERT_EQUAL(*iter0, T{});
}
DECLARE_GENERIC_UNITTEST(TestCountingDefaultConstructor);

void TestCountingIteratorCopyConstructor()
{
  thrust::counting_iterator<int> iter0(100);

  thrust::counting_iterator<int> iter1(iter0);

  ASSERT_EQUAL_QUIET(iter0, iter1);
  ASSERT_EQUAL(*iter0, *iter1);

  // construct from related space
  thrust::counting_iterator<int, thrust::host_system_tag> h_iter = iter0;
  ASSERT_EQUAL(*iter0, *h_iter);

  thrust::counting_iterator<int, thrust::device_system_tag> d_iter = iter0;
  ASSERT_EQUAL(*iter0, *d_iter);
}
DECLARE_UNITTEST(TestCountingIteratorCopyConstructor);
static_assert(cuda::std::is_trivially_copy_constructible<thrust::counting_iterator<int>>::value, "");
static_assert(cuda::std::is_trivially_copyable<thrust::counting_iterator<int>>::value, "");

void TestCountingIteratorIncrement()
{
  thrust::counting_iterator<int> iter(0);

  ASSERT_EQUAL(*iter, 0);

  iter++;

  ASSERT_EQUAL(*iter, 1);

  iter++;
  iter++;

  ASSERT_EQUAL(*iter, 3);

  iter += 5;

  ASSERT_EQUAL(*iter, 8);

  iter -= 10;

  ASSERT_EQUAL(*iter, -2);
}
DECLARE_UNITTEST(TestCountingIteratorIncrement);

void TestCountingIteratorComparison()
{
  thrust::counting_iterator<int> iter1(0);
  thrust::counting_iterator<int> iter2(0);

  ASSERT_EQUAL(iter1 - iter2, 0);
  ASSERT_EQUAL(iter1 == iter2, true);

  iter1++;

  ASSERT_EQUAL(iter1 - iter2, 1);
  ASSERT_EQUAL(iter1 == iter2, false);

  iter2++;

  ASSERT_EQUAL(iter1 - iter2, 0);
  ASSERT_EQUAL(iter1 == iter2, true);

  iter1 += 100;
  iter2 += 100;

  ASSERT_EQUAL(iter1 - iter2, 0);
  ASSERT_EQUAL(iter1 == iter2, true);
}
DECLARE_UNITTEST(TestCountingIteratorComparison);

void TestCountingIteratorFloatComparison()
{
  thrust::counting_iterator<float> iter1(0);
  thrust::counting_iterator<float> iter2(0);

  ASSERT_EQUAL(iter1 - iter2, 0);
  ASSERT_EQUAL(iter1 == iter2, true);
  ASSERT_EQUAL(iter1 < iter2, false);
  ASSERT_EQUAL(iter2 < iter1, false);

  iter1++;

  ASSERT_EQUAL(iter1 - iter2, 1);
  ASSERT_EQUAL(iter1 == iter2, false);
  ASSERT_EQUAL(iter2 < iter1, true);
  ASSERT_EQUAL(iter1 < iter2, false);

  iter2++;

  ASSERT_EQUAL(iter1 - iter2, 0);
  ASSERT_EQUAL(iter1 == iter2, true);
  ASSERT_EQUAL(iter1 < iter2, false);
  ASSERT_EQUAL(iter2 < iter1, false);

  iter1 += 100;
  iter2 += 100;

  ASSERT_EQUAL(iter1 - iter2, 0);
  ASSERT_EQUAL(iter1 == iter2, true);
  ASSERT_EQUAL(iter1 < iter2, false);
  ASSERT_EQUAL(iter2 < iter1, false);

  thrust::counting_iterator<float> iter3(0);
  thrust::counting_iterator<float> iter4(0.5);

  ASSERT_EQUAL(iter3 - iter4, 0);
  ASSERT_EQUAL(iter3 == iter4, true);
  ASSERT_EQUAL(iter3 < iter4, false);
  ASSERT_EQUAL(iter4 < iter3, false);

  iter3++; // iter3 = 1.0, iter4 = 0.5

  ASSERT_EQUAL(iter3 - iter4, 0);
  ASSERT_EQUAL(iter3 == iter4, true);
  ASSERT_EQUAL(iter3 < iter4, false);
  ASSERT_EQUAL(iter4 < iter3, false);

  iter4++; // iter3 = 1.0, iter4 = 1.5

  ASSERT_EQUAL(iter3 - iter4, 0);
  ASSERT_EQUAL(iter3 == iter4, true);
  ASSERT_EQUAL(iter3 < iter4, false);
  ASSERT_EQUAL(iter4 < iter3, false);

  iter4++; // iter3 = 1.0, iter4 = 2.5

  ASSERT_EQUAL(iter3 - iter4, -1);
  ASSERT_EQUAL(iter4 - iter3, 1);
  ASSERT_EQUAL(iter3 == iter4, false);
  ASSERT_EQUAL(iter3 < iter4, true);
  ASSERT_EQUAL(iter4 < iter3, false);
}
DECLARE_UNITTEST(TestCountingIteratorFloatComparison);

void TestCountingIteratorDistance()
{
  thrust::counting_iterator<int> iter1(0);
  thrust::counting_iterator<int> iter2(5);

  ASSERT_EQUAL(thrust::distance(iter1, iter2), 5);

  iter1++;

  ASSERT_EQUAL(thrust::distance(iter1, iter2), 4);

  iter2 += 100;

  ASSERT_EQUAL(thrust::distance(iter1, iter2), 104);
}
DECLARE_UNITTEST(TestCountingIteratorDistance);

void TestCountingIteratorUnsignedType()
{
  thrust::counting_iterator<unsigned int> iter0(0);
  thrust::counting_iterator<unsigned int> iter1(5);

  ASSERT_EQUAL(iter1 - iter0, 5);
  ASSERT_EQUAL(iter0 - iter1, -5);
  ASSERT_EQUAL(iter0 != iter1, true);
  ASSERT_EQUAL(iter0 < iter1, true);
  ASSERT_EQUAL(iter1 < iter0, false);
}
DECLARE_UNITTEST(TestCountingIteratorUnsignedType);

void TestCountingIteratorLowerBound()
{
  size_t n       = 10000;
  const size_t M = 100;

  thrust::host_vector<unsigned int> h_data = unittest::random_integers<unsigned int>(n);
  for (unsigned int i = 0; i < n; ++i)
  {
    h_data[i] %= M;
  }

  thrust::sort(h_data.begin(), h_data.end());

  thrust::device_vector<unsigned int> d_data = h_data;

  thrust::counting_iterator<unsigned int> search_begin(0);
  thrust::counting_iterator<unsigned int> search_end(M);

  thrust::host_vector<unsigned int> h_result(M);
  thrust::device_vector<unsigned int> d_result(M);

  thrust::lower_bound(h_data.begin(), h_data.end(), search_begin, search_end, h_result.begin());

  thrust::lower_bound(d_data.begin(), d_data.end(), search_begin, search_end, d_result.begin());

  ASSERT_EQUAL(h_result, d_result);
}
DECLARE_UNITTEST(TestCountingIteratorLowerBound);

void TestCountingIteratorDifference()
{
  using Iterator   = thrust::counting_iterator<std::uint64_t>;
  using Difference = thrust::iterator_difference<Iterator>::type;

  Difference diff = std::numeric_limits<std::uint32_t>::max() + 1;

  Iterator first(0);
  Iterator last = first + diff;

  ASSERT_EQUAL(diff, last - first);
}
DECLARE_UNITTEST(TestCountingIteratorDifference);

_CCCL_DIAG_POP
