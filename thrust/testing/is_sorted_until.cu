#include <thrust/iterator/retag.h>
#include <thrust/sort.h>

#include <unittest/unittest.h>

template <typename Vector>
void TestIsSortedUntilSimple()
{
  using T        = typename Vector::value_type;
  using Iterator = typename Vector::iterator;

  Vector v{0, 5, 8, 0};

  Iterator first = v.begin();

  Iterator last = v.begin() + 0;
  Iterator ref  = last;
  ASSERT_EQUAL_QUIET(ref, thrust::is_sorted_until(first, last));

  last = v.begin() + 1;
  ref  = last;
  ASSERT_EQUAL_QUIET(ref, thrust::is_sorted_until(first, last));

  last = v.begin() + 2;
  ref  = last;
  ASSERT_EQUAL_QUIET(ref, thrust::is_sorted_until(first, last));

  last = v.begin() + 3;
  ref  = v.begin() + 3;
  ASSERT_EQUAL_QUIET(ref, thrust::is_sorted_until(first, last));

  last = v.begin() + 4;
  ref  = v.begin() + 3;
  ASSERT_EQUAL_QUIET(ref, thrust::is_sorted_until(first, last));

  last = v.begin() + 3;
  ref  = v.begin() + 3;
  ASSERT_EQUAL_QUIET(ref, thrust::is_sorted_until(first, last, thrust::less<T>()));

  last = v.begin() + 4;
  ref  = v.begin() + 3;
  ASSERT_EQUAL_QUIET(ref, thrust::is_sorted_until(first, last, thrust::less<T>()));

  last = v.begin() + 1;
  ref  = v.begin() + 1;
  ASSERT_EQUAL_QUIET(ref, thrust::is_sorted_until(first, last, thrust::greater<T>()));

  last = v.begin() + 4;
  ref  = v.begin() + 1;
  ASSERT_EQUAL_QUIET(ref, thrust::is_sorted_until(first, last, thrust::greater<T>()));

  first = v.begin() + 2;
  last  = v.begin() + 4;
  ref   = v.begin() + 4;
  ASSERT_EQUAL_QUIET(ref, thrust::is_sorted_until(first, last, thrust::greater<T>()));
}
DECLARE_VECTOR_UNITTEST(TestIsSortedUntilSimple);

template <typename Vector>
void TestIsSortedUntilRepeatedElements()
{
  Vector v{0, 1, 1, 2, 3, 4, 5, 5, 5, 6};

  ASSERT_EQUAL_QUIET(v.end(), thrust::is_sorted_until(v.begin(), v.end()));
}
DECLARE_VECTOR_UNITTEST(TestIsSortedUntilRepeatedElements);

template <class Vector>
void TestIsSortedUntil()
{
  using T = typename Vector::value_type;

  const size_t n = (1 << 16) + 13;

  Vector v = unittest::random_integers<T>(n);

  v[0] = 1;
  v[1] = 0;

  ASSERT_EQUAL_QUIET(v.begin() + 1, thrust::is_sorted_until(v.begin(), v.end()));

  thrust::sort(v.begin(), v.end());

  ASSERT_EQUAL_QUIET(v.end(), thrust::is_sorted_until(v.begin(), v.end()));
}
DECLARE_INTEGRAL_VECTOR_UNITTEST(TestIsSortedUntil);

template <typename ForwardIterator>
ForwardIterator is_sorted_until(my_system& system, ForwardIterator first, ForwardIterator)
{
  system.validate_dispatch();
  return first;
}

void TestIsSortedUntilExplicit()
{
  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::is_sorted_until(sys, vec.begin(), vec.end());

  ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestIsSortedUntilExplicit);

template <typename ForwardIterator>
ForwardIterator is_sorted_until(my_tag, ForwardIterator first, ForwardIterator)
{
  *first = 13;
  return first;
}

void TestIsSortedUntilImplicit()
{
  thrust::device_vector<int> vec(1);

  thrust::is_sorted_until(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.end()));

  ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestIsSortedUntilImplicit);
