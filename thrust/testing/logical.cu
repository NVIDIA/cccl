#include <thrust/functional.h>
#include <thrust/iterator/retag.h>
#include <thrust/logical.h>

#include <unittest/unittest.h>

template <class Vector>
void TestAllOf()
{
  using T = typename Vector::value_type;

  Vector v(3, T{1});

  REQUIRE(thrust::all_of(v.begin(), v.end(), ::cuda::std::identity{}));

  v[1] = T{0};

  REQUIRE_FALSE(thrust::all_of(v.begin(), v.end(), ::cuda::std::identity{}));

  REQUIRE(thrust::all_of(v.begin() + 0, v.begin() + 0, ::cuda::std::identity{}));
  REQUIRE(thrust::all_of(v.begin() + 0, v.begin() + 1, ::cuda::std::identity{}));
  REQUIRE_FALSE(thrust::all_of(v.begin() + 0, v.begin() + 2, ::cuda::std::identity{}));
  REQUIRE_FALSE(thrust::all_of(v.begin() + 1, v.begin() + 2, ::cuda::std::identity{}));
}
DECLARE_VECTOR_UNITTEST(TestAllOf);

template <class InputIterator, class Predicate>
bool all_of(my_system& system, InputIterator, InputIterator, Predicate)
{
  system.validate_dispatch();
  return false;
}

TEST_CASE("TestAllOfDispatchExplicit", "[logical]")
{
  thrust::device_vector<int> vec(1);

  my_system sys(0); // NOLINT(misc-const-correctness)
  thrust::all_of(sys, vec.begin(), vec.end(), 0);

  REQUIRE(sys.is_valid());
}

template <class InputIterator, class Predicate>
bool all_of(my_tag, InputIterator first, InputIterator, Predicate)
{
  *first = 13;
  return false;
}

TEST_CASE("TestAllOfDispatchImplicit", "[logical]")
{
  thrust::device_vector<int> vec(1);

  thrust::all_of(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.end()), 0);

  REQUIRE(13 == vec.front());
}

template <class Vector>
void TestAnyOf()
{
  using T = typename Vector::value_type;

  Vector v(3, T{1});

  REQUIRE(thrust::any_of(v.begin(), v.end(), ::cuda::std::identity{}));

  v[1] = 0;

  REQUIRE(thrust::any_of(v.begin(), v.end(), ::cuda::std::identity{}));

  REQUIRE_FALSE(thrust::any_of(v.begin() + 0, v.begin() + 0, ::cuda::std::identity{}));
  REQUIRE(thrust::any_of(v.begin() + 0, v.begin() + 1, ::cuda::std::identity{}));
  REQUIRE(thrust::any_of(v.begin() + 0, v.begin() + 2, ::cuda::std::identity{}));
  REQUIRE_FALSE(thrust::any_of(v.begin() + 1, v.begin() + 2, ::cuda::std::identity{}));
}
DECLARE_VECTOR_UNITTEST(TestAnyOf);

template <class InputIterator, class Predicate>
bool any_of(my_system& system, InputIterator, InputIterator, Predicate)
{
  system.validate_dispatch();
  return false;
}

TEST_CASE("TestAnyOfDispatchExplicit", "[logical]")
{
  thrust::device_vector<int> vec(1);

  my_system sys(0); // NOLINT(misc-const-correctness)
  thrust::any_of(sys, vec.begin(), vec.end(), 0);

  REQUIRE(sys.is_valid());
}

template <class InputIterator, class Predicate>
bool any_of(my_tag, InputIterator first, InputIterator, Predicate)
{
  *first = 13;
  return false;
}

TEST_CASE("TestAnyOfDispatchImplicit", "[logical]")
{
  thrust::device_vector<int> vec(1);

  thrust::any_of(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.end()), 0);

  REQUIRE(13 == vec.front());
}

template <class Vector>
void TestNoneOf()
{
  using T = typename Vector::value_type;

  Vector v(3, T{1});

  REQUIRE_FALSE(thrust::none_of(v.begin(), v.end(), ::cuda::std::identity{}));

  v[1] = 0;

  REQUIRE_FALSE(thrust::none_of(v.begin(), v.end(), ::cuda::std::identity{}));

  REQUIRE(thrust::none_of(v.begin() + 0, v.begin() + 0, ::cuda::std::identity{}));
  REQUIRE_FALSE(thrust::none_of(v.begin() + 0, v.begin() + 1, ::cuda::std::identity{}));
  REQUIRE_FALSE(thrust::none_of(v.begin() + 0, v.begin() + 2, ::cuda::std::identity{}));
  REQUIRE(thrust::none_of(v.begin() + 1, v.begin() + 2, ::cuda::std::identity{}));
}
DECLARE_VECTOR_UNITTEST(TestNoneOf);

template <class InputIterator, class Predicate>
bool none_of(my_system& system, InputIterator, InputIterator, Predicate)
{
  system.validate_dispatch();
  return false;
}

TEST_CASE("TestNoneOfDispatchExplicit", "[logical]")
{
  thrust::device_vector<int> vec(1);

  my_system sys(0); // NOLINT(misc-const-correctness)
  thrust::none_of(sys, vec.begin(), vec.end(), 0);

  REQUIRE(sys.is_valid());
}

template <class InputIterator, class Predicate>
bool none_of(my_tag, InputIterator first, InputIterator, Predicate)
{
  *first = 13;
  return false;
}

TEST_CASE("TestNoneOfDispatchImplicit", "[logical]")
{
  thrust::device_vector<int> vec(1);

  thrust::none_of(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.end()), 0);

  REQUIRE(13 == vec.front());
}
