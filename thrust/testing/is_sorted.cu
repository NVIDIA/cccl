#include <thrust/iterator/retag.h>
#include <thrust/sort.h>

#include <unittest/unittest.h>

template <class Vector>
void TestIsSortedSimple()
{
  using T = typename Vector::value_type;

  Vector v{0, 5, 8, 0};

  REQUIRE(thrust::is_sorted(v.begin(), v.begin() + 0));
  REQUIRE(thrust::is_sorted(v.begin(), v.begin() + 1));

  // the following line crashes gcc 4.3
#if (__GNUC__ == 4) && (__GNUC_MINOR__ == 3)
  // do nothing
#else
  // compile this line on other compilers
  REQUIRE(thrust::is_sorted(v.begin(), v.begin() + 2));
#endif // GCC

  REQUIRE(thrust::is_sorted(v.begin(), v.begin() + 3));
  REQUIRE_FALSE(thrust::is_sorted(v.begin(), v.begin() + 4));

  REQUIRE(thrust::is_sorted(v.begin(), v.begin() + 3, ::cuda::std::less<T>()));

  REQUIRE(thrust::is_sorted(v.begin(), v.begin() + 1, ::cuda::std::greater<T>()));
  REQUIRE_FALSE(thrust::is_sorted(v.begin(), v.begin() + 4, ::cuda::std::greater<T>()));

  REQUIRE_FALSE(thrust::is_sorted(v.begin(), v.end()));
}
DECLARE_VECTOR_UNITTEST(TestIsSortedSimple);

template <class Vector>
void TestIsSortedRepeatedElements()
{
  Vector v{0, 1, 1, 2, 3, 4, 5, 5, 5, 6};

  REQUIRE(thrust::is_sorted(v.begin(), v.end()));
}
DECLARE_VECTOR_UNITTEST(TestIsSortedRepeatedElements);

template <class Vector>
void TestIsSorted()
{
  using T = typename Vector::value_type;

  const size_t n = (1 << 16) + 13;

  Vector v = unittest::random_integers<T>(n);

  v[0] = 1;
  v[1] = 0;

  REQUIRE_FALSE(thrust::is_sorted(v.begin(), v.end()));

  thrust::sort(v.begin(), v.end());

  REQUIRE(thrust::is_sorted(v.begin(), v.end()));
}
DECLARE_INTEGRAL_VECTOR_UNITTEST(TestIsSorted);

template <typename InputIterator>
bool is_sorted(my_system& system, InputIterator /*first*/, InputIterator)
{
  system.validate_dispatch();
  return false;
}

void TestIsSortedDispatchExplicit()
{
  thrust::device_vector<int> vec(1);

  my_system sys(0); // NOLINT(misc-const-correctness)
  thrust::is_sorted(sys, vec.begin(), vec.end());

  REQUIRE(sys.is_valid());
}
TEST_CASE("TestIsSortedDispatchExplicit", "[is_sorted]")
{
  TestIsSortedDispatchExplicit();
}

template <typename InputIterator>
bool is_sorted(my_tag, InputIterator first, InputIterator)
{
  *first = 13;
  return false;
}

void TestIsSortedDispatchImplicit()
{
  thrust::device_vector<int> vec(1);

  thrust::is_sorted(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.end()));

  REQUIRE(13 == vec.front());
}
TEST_CASE("TestIsSortedDispatchImplicit", "[is_sorted]")
{
  TestIsSortedDispatchImplicit();
}
