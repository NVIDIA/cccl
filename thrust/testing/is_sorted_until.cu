#include <thrust/iterator/retag.h>
#include <thrust/sort.h>

#include <unittest/unittest.h>

template <typename Vector>
void test_is_sorted_until_simple()
{
  using T        = typename Vector::value_type;
  using Iterator = typename Vector::iterator;

  Vector v{0, 5, 8, 0};

  Iterator first = v.begin();

  Iterator last = v.begin() + 0;
  Iterator ref  = last;
  REQUIRE(ref == thrust::is_sorted_until(first, last));

  last = v.begin() + 1;
  ref  = last;
  REQUIRE(ref == thrust::is_sorted_until(first, last));

  last = v.begin() + 2;
  ref  = last;
  REQUIRE(ref == thrust::is_sorted_until(first, last));

  last = v.begin() + 3;
  ref  = v.begin() + 3;
  REQUIRE(ref == thrust::is_sorted_until(first, last));

  last = v.begin() + 4;
  ref  = v.begin() + 3;
  REQUIRE(ref == thrust::is_sorted_until(first, last));

  last = v.begin() + 3;
  ref  = v.begin() + 3;
  REQUIRE(ref == thrust::is_sorted_until(first, last, ::cuda::std::less<T>()));

  last = v.begin() + 4;
  ref  = v.begin() + 3;
  REQUIRE(ref == thrust::is_sorted_until(first, last, ::cuda::std::less<T>()));

  last = v.begin() + 1;
  ref  = v.begin() + 1;
  REQUIRE(ref == thrust::is_sorted_until(first, last, ::cuda::std::greater<T>()));

  last = v.begin() + 4;
  ref  = v.begin() + 1;
  REQUIRE(ref == thrust::is_sorted_until(first, last, ::cuda::std::greater<T>()));

  first = v.begin() + 2;
  last  = v.begin() + 4;
  ref   = v.begin() + 4;
  REQUIRE(ref == thrust::is_sorted_until(first, last, ::cuda::std::greater<T>()));
}
DECLARE_VECTOR_UNITTEST(test_is_sorted_until_simple);

template <typename Vector>
void test_is_sorted_until_repeated_elements()
{
  Vector v{0, 1, 1, 2, 3, 4, 5, 5, 5, 6};

  REQUIRE(v.end() == thrust::is_sorted_until(v.begin(), v.end()));
}
DECLARE_VECTOR_UNITTEST(test_is_sorted_until_repeated_elements);

template <class Vector>
void test_is_sorted_until()
{
  using T = typename Vector::value_type;

  const size_t n = (1 << 16) + 13;

  Vector v = unittest::random_integers<T>(n);

  v[0] = 1;
  v[1] = 0;

  REQUIRE(v.begin() + 1 == thrust::is_sorted_until(v.begin(), v.end()));

  thrust::sort(v.begin(), v.end());

  REQUIRE(v.end() == thrust::is_sorted_until(v.begin(), v.end()));
}
DECLARE_INTEGRAL_VECTOR_UNITTEST(test_is_sorted_until);

template <typename ForwardIterator>
ForwardIterator is_sorted_until(my_system& system, ForwardIterator first, ForwardIterator)
{
  system.validate_dispatch();
  return first;
}

TEST_CASE("TestIsSortedUntilExplicit", "[is_sorted_until]")
{
  thrust::device_vector<int> vec(1);

  my_system sys(0); // NOLINT(misc-const-correctness)
  thrust::is_sorted_until(sys, vec.begin(), vec.end());

  REQUIRE(sys.is_valid());
}

template <typename ForwardIterator>
ForwardIterator is_sorted_until(my_tag, ForwardIterator first, ForwardIterator)
{
  *first = 13;
  return first;
}

TEST_CASE("TestIsSortedUntilImplicit", "[is_sorted_until]")
{
  thrust::device_vector<int> vec(1);

  thrust::is_sorted_until(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.end()));

  REQUIRE(13 == vec.front());
}
