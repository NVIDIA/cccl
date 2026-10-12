#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/retag.h>
#include <thrust/replace.h>

#include <unittest/unittest.h>

// New GCC, new miscompile. 13 + TBB this time.
#if _CCCL_COMPILER(GCC, ==, 13) && THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_TBB
#  define THRUST_GCC13_TBB_MISCOMPILE
#endif

template <class Vector>
void test_replace_simple()
{
  using T = typename Vector::value_type;

  Vector data{1, 2, 1, 3, 2};

  thrust::replace(data.begin(), data.end(), (T) 1, (T) 4);
  thrust::replace(data.begin(), data.end(), (T) 2, (T) 5);

  Vector result{4, 5, 4, 3, 5};

  REQUIRE(data == result);
}
DECLARE_VECTOR_UNITTEST(test_replace_simple);

template <typename ForwardIterator, typename T>
void replace(my_system& system, ForwardIterator, ForwardIterator, const T&, const T&)
{
  system.validate_dispatch();
}

TEST_CASE("TestReplaceDispatchExplicit", "[replace]")
{
  thrust::device_vector<int> vec(1);

  my_system sys(0); // NOLINT(misc-const-correctness)
  thrust::replace(sys, vec.begin(), vec.begin(), 0, 0);

  REQUIRE(sys.is_valid());
}

template <typename ForwardIterator, typename T>
void replace(my_tag, ForwardIterator first, ForwardIterator, const T&, const T&)
{
  *first = 13;
}

TEST_CASE("TestReplaceDispatchImplicit", "[replace]")
{
  thrust::device_vector<int> vec(1);

  thrust::replace(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()), 0, 0);

  REQUIRE(13 == vec.front());
}

template <typename T>
void test_replace(const size_t n)
{
  thrust::host_vector<T> h_data   = unittest::random_samples<T>(n);
  thrust::device_vector<T> d_data = h_data;

  T old_value = 0;
  T new_value = 1;

  thrust::replace(h_data.begin(), h_data.end(), old_value, new_value);
  thrust::replace(d_data.begin(), d_data.end(), old_value, new_value);

  ASSERT_ALMOST_EQUAL(h_data, d_data);
}
DECLARE_VARIABLE_UNITTEST(test_replace);

#ifndef THRUST_GCC13_TBB_MISCOMPILE
template <class Vector>
THRUST_DISABLE_BROKEN_GCC_VECTORIZER void test_replace_copy_simple()
{
  using T = typename Vector::value_type;

  Vector data{1, 2, 1, 3, 2};

  Vector dest(5);

  thrust::replace_copy(data.begin(), data.end(), dest.begin(), (T) 1, (T) 4);
  thrust::replace_copy(dest.begin(), dest.end(), dest.begin(), (T) 2, (T) 5);

  Vector result{4, 5, 4, 3, 5};
  REQUIRE(dest == result);
}
DECLARE_VECTOR_UNITTEST(test_replace_copy_simple);
#endif

template <typename InputIterator, typename OutputIterator, typename T>
OutputIterator replace_copy(my_system& system, InputIterator, InputIterator, OutputIterator result, const T&, const T&)
{
  system.validate_dispatch();
  return result;
}

TEST_CASE("TestReplaceCopyDispatchExplicit", "[replace]")
{
  thrust::device_vector<int> vec(1);

  my_system sys(0); // NOLINT(misc-const-correctness)
  thrust::replace_copy(sys, vec.begin(), vec.begin(), vec.begin(), 0, 0);

  REQUIRE(sys.is_valid());
}

template <typename InputIterator, typename OutputIterator, typename T>
OutputIterator replace_copy(my_tag, InputIterator, InputIterator, OutputIterator result, const T&, const T&)
{
  *result = 13;
  return result;
}

TEST_CASE("TestReplaceCopyDispatchImplicit", "[replace]")
{
  thrust::device_vector<int> vec(1);

  thrust::replace_copy(
    thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()), 0, 0);

  REQUIRE(13 == vec.front());
}

template <typename T>
void test_replace_copy(const size_t n)
{
  thrust::host_vector<T> h_data   = unittest::random_samples<T>(n);
  thrust::device_vector<T> d_data = h_data;

  T old_value = 0;
  T new_value = 1;

  thrust::host_vector<T> h_dest(n);
  thrust::device_vector<T> d_dest(n);

  thrust::replace_copy(h_data.begin(), h_data.end(), h_dest.begin(), old_value, new_value);
  thrust::replace_copy(d_data.begin(), d_data.end(), d_dest.begin(), old_value, new_value);

  ASSERT_ALMOST_EQUAL(h_data, d_data);
  ASSERT_ALMOST_EQUAL(h_dest, d_dest);
}
DECLARE_VARIABLE_UNITTEST(test_replace_copy);

template <typename T>
void test_replace_copy_to_discard_iterator(const size_t n)
{
  thrust::host_vector<T> h_data   = unittest::random_samples<T>(n);
  thrust::device_vector<T> d_data = h_data;

  T old_value = 0;
  T new_value = 1;

  const thrust::discard_iterator<> h_result =
    thrust::replace_copy(h_data.begin(), h_data.end(), thrust::make_discard_iterator(), old_value, new_value);

  const thrust::discard_iterator<> d_result =
    thrust::replace_copy(d_data.begin(), d_data.end(), thrust::make_discard_iterator(), old_value, new_value);

  const thrust::discard_iterator<> reference(static_cast<std::ptrdiff_t>(n));

  REQUIRE(reference == h_result);
  REQUIRE(reference == d_result);
}
DECLARE_VARIABLE_UNITTEST(test_replace_copy_to_discard_iterator);

template <typename T>
struct less_than_five
{
  _CCCL_HOST_DEVICE bool operator()(const T& val) const
  {
    return val < 5;
  }
};

template <class Vector>
void test_replace_if_simple()
{
  using T = typename Vector::value_type;

  Vector data{1, 3, 4, 6, 5};

  thrust::replace_if(data.begin(), data.end(), less_than_five<T>(), (T) 0);

  Vector result{0, 0, 0, 6, 5};

  REQUIRE(data == result);
}
DECLARE_VECTOR_UNITTEST(test_replace_if_simple);

template <typename ForwardIterator, typename Predicate, typename T>
void replace_if(my_system& system, ForwardIterator, ForwardIterator, Predicate, const T&)
{
  system.validate_dispatch();
}

TEST_CASE("TestReplaceIfDispatchExplicit", "[replace]")
{
  thrust::device_vector<int> vec(1);

  my_system sys(0); // NOLINT(misc-const-correctness)
  thrust::replace_if(sys, vec.begin(), vec.begin(), 0, 0);

  REQUIRE(sys.is_valid());
}

template <typename ForwardIterator, typename Predicate, typename T>
void replace_if(my_tag, ForwardIterator first, ForwardIterator, Predicate, const T&)
{
  *first = 13;
}

TEST_CASE("TestReplaceIfDispatchImplicit", "[replace]")
{
  thrust::device_vector<int> vec(1);

  thrust::replace_if(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()), 0, 0);

  REQUIRE(13 == vec.front());
}

template <class Vector>
THRUST_DISABLE_BROKEN_GCC_VECTORIZER void test_replace_if_stencil_simple()
{
  using T = typename Vector::value_type;

  Vector data{1, 3, 4, 6, 5};

  Vector stencil{5, 4, 6, 3, 7};
  thrust::replace_if(data.begin(), data.end(), stencil.begin(), less_than_five<T>(), (T) 0);

  Vector result{1, 0, 4, 0, 5};

  REQUIRE(data == result);
}
DECLARE_VECTOR_UNITTEST(test_replace_if_stencil_simple);

template <typename ForwardIterator, typename InputIterator, typename Predicate, typename T>
void replace_if(my_system& system, ForwardIterator, ForwardIterator, InputIterator, Predicate, const T&)
{
  system.validate_dispatch();
}

TEST_CASE("TestReplaceIfStencilDispatchExplicit", "[replace]")
{
  thrust::device_vector<int> vec(1);

  my_system sys(0); // NOLINT(misc-const-correctness)
  thrust::replace_if(sys, vec.begin(), vec.begin(), vec.begin(), 0, 0);

  REQUIRE(sys.is_valid());
}

template <typename ForwardIterator, typename InputIterator, typename Predicate, typename T>
void replace_if(my_tag, ForwardIterator first, ForwardIterator, InputIterator, Predicate, const T&)
{
  *first = 13;
}

TEST_CASE("TestReplaceIfStencilDispatchImplicit", "[replace]")
{
  thrust::device_vector<int> vec(1);

  thrust::replace_if(
    thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()), 0, 0);

  REQUIRE(13 == vec.front());
}

template <typename T>
THRUST_DISABLE_BROKEN_GCC_VECTORIZER void test_replace_if(const size_t n)
{
  thrust::host_vector<T> h_data   = unittest::random_samples<T>(n);
  thrust::device_vector<T> d_data = h_data;

  thrust::replace_if(h_data.begin(), h_data.end(), less_than_five<T>(), (T) 0);
  thrust::replace_if(d_data.begin(), d_data.end(), less_than_five<T>(), (T) 0);

  ASSERT_ALMOST_EQUAL(h_data, d_data);
}
DECLARE_VARIABLE_UNITTEST(test_replace_if);

template <typename T>
THRUST_DISABLE_BROKEN_GCC_VECTORIZER void test_replace_if_stencil(const size_t n)
{
  thrust::host_vector<T> h_data   = unittest::random_samples<T>(n);
  thrust::device_vector<T> d_data = h_data;

  thrust::host_vector<T> h_stencil   = unittest::random_samples<T>(n);
  thrust::device_vector<T> d_stencil = h_stencil;

  thrust::replace_if(h_data.begin(), h_data.end(), h_stencil.begin(), less_than_five<T>(), (T) 0);
  thrust::replace_if(d_data.begin(), d_data.end(), d_stencil.begin(), less_than_five<T>(), (T) 0);

  ASSERT_ALMOST_EQUAL(h_data, d_data);
}
DECLARE_VARIABLE_UNITTEST(test_replace_if_stencil);

template <class Vector>
THRUST_DISABLE_BROKEN_GCC_VECTORIZER void test_replace_copy_if_simple()
{
  using T = typename Vector::value_type;

  Vector data{1, 3, 4, 6, 5};

  Vector dest(5);

  thrust::replace_copy_if(data.begin(), data.end(), dest.begin(), less_than_five<T>(), (T) 0);

  Vector result{0, 0, 0, 6, 5};
  REQUIRE(dest == result);
}
DECLARE_VECTOR_UNITTEST(test_replace_copy_if_simple);

template <typename InputIterator, typename OutputIterator, typename Predicate, typename T>
OutputIterator
replace_copy_if(my_system& system, InputIterator, InputIterator, OutputIterator result, Predicate, const T&)
{
  system.validate_dispatch();
  return result;
}

TEST_CASE("TestReplaceCopyIfDispatchExplicit", "[replace]")
{
  thrust::device_vector<int> vec(1);

  my_system sys(0); // NOLINT(misc-const-correctness)
  thrust::replace_copy_if(sys, vec.begin(), vec.begin(), vec.begin(), 0, 0);

  REQUIRE(sys.is_valid());
}

template <typename InputIterator, typename OutputIterator, typename Predicate, typename T>
OutputIterator replace_copy_if(my_tag, InputIterator, InputIterator, OutputIterator result, Predicate, const T&)
{
  *result = 13;
  return result;
}

TEST_CASE("TestReplaceCopyIfDispatchImplicit", "[replace]")
{
  thrust::device_vector<int> vec(1);

  thrust::replace_copy_if(
    thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()), 0, 0);

  REQUIRE(13 == vec.front());
}

template <class Vector>
THRUST_DISABLE_BROKEN_GCC_VECTORIZER void test_replace_copy_if_stencil_simple()
{
  using T = typename Vector::value_type;

  Vector data{1, 3, 4, 6, 5};
  Vector stencil{1, 5, 4, 7, 8};

  Vector dest(5);

  thrust::replace_copy_if(data.begin(), data.end(), stencil.begin(), dest.begin(), less_than_five<T>(), (T) 0);

  Vector result{0, 3, 0, 6, 5};

  REQUIRE(dest == result);
}
DECLARE_VECTOR_UNITTEST(test_replace_copy_if_stencil_simple);

template <typename InputIterator1, typename InputIterator2, typename OutputIterator, typename Predicate, typename T>
OutputIterator replace_copy_if(
  my_system& system, InputIterator1, InputIterator1, InputIterator2, OutputIterator result, Predicate, const T&)
{
  system.validate_dispatch();
  return result;
}

TEST_CASE("TestReplaceCopyIfStencilDispatchExplicit", "[replace]")
{
  thrust::device_vector<int> vec(1);

  my_system sys(0); // NOLINT(misc-const-correctness)
  thrust::replace_copy_if(sys, vec.begin(), vec.begin(), vec.begin(), vec.begin(), 0, 0);

  REQUIRE(sys.is_valid());
}

template <typename InputIterator1, typename InputIterator2, typename OutputIterator, typename Predicate, typename T>
OutputIterator
replace_copy_if(my_tag, InputIterator1, InputIterator1, InputIterator2, OutputIterator result, Predicate, const T&)
{
  *result = 13;
  return result;
}

TEST_CASE("TestReplaceCopyIfStencilDispatchImplicit", "[replace]")
{
  thrust::device_vector<int> vec(1);

  thrust::replace_copy_if(
    thrust::retag<my_tag>(vec.begin()),
    thrust::retag<my_tag>(vec.begin()),
    thrust::retag<my_tag>(vec.begin()),
    thrust::retag<my_tag>(vec.begin()),
    0,
    0);

  REQUIRE(13 == vec.front());
}

template <typename T>
THRUST_DISABLE_BROKEN_GCC_VECTORIZER void test_replace_copy_if(const size_t n)
{
  thrust::host_vector<T> h_data   = unittest::random_samples<T>(n);
  thrust::device_vector<T> d_data = h_data;

  thrust::host_vector<T> h_dest(n);
  thrust::device_vector<T> d_dest(n);

  thrust::replace_copy_if(h_data.begin(), h_data.end(), h_dest.begin(), less_than_five<T>(), T{0});
  thrust::replace_copy_if(d_data.begin(), d_data.end(), d_dest.begin(), less_than_five<T>(), T{0});

  ASSERT_ALMOST_EQUAL(h_data, d_data);
  ASSERT_ALMOST_EQUAL(h_dest, d_dest);
}
DECLARE_VARIABLE_UNITTEST(test_replace_copy_if);

template <typename T>
THRUST_DISABLE_BROKEN_GCC_VECTORIZER void test_replace_copy_if_to_discard_iterator(const size_t n)
{
  thrust::host_vector<T> h_data   = unittest::random_samples<T>(n);
  thrust::device_vector<T> d_data = h_data;

  const thrust::discard_iterator<> h_result =
    thrust::replace_copy_if(h_data.begin(), h_data.end(), thrust::make_discard_iterator(), less_than_five<T>(), T{0});

  const thrust::discard_iterator<> d_result =
    thrust::replace_copy_if(d_data.begin(), d_data.end(), thrust::make_discard_iterator(), less_than_five<T>(), T{0});

  const thrust::discard_iterator<> reference(static_cast<std::ptrdiff_t>(n));

  REQUIRE(reference == h_result);
  REQUIRE(reference == d_result);
}
DECLARE_VARIABLE_UNITTEST(test_replace_copy_if_to_discard_iterator);

template <typename T>
THRUST_DISABLE_BROKEN_GCC_VECTORIZER void test_replace_copy_if_stencil(const size_t n)
{
  thrust::host_vector<T> h_data   = unittest::random_samples<T>(n);
  thrust::device_vector<T> d_data = h_data;

  thrust::host_vector<T> h_stencil   = unittest::random_samples<T>(n);
  thrust::device_vector<T> d_stencil = h_stencil;

  thrust::host_vector<T> h_dest(n);
  thrust::device_vector<T> d_dest(n);

  thrust::replace_copy_if(h_data.begin(), h_data.end(), h_stencil.begin(), h_dest.begin(), less_than_five<T>(), T{0});
  thrust::replace_copy_if(d_data.begin(), d_data.end(), d_stencil.begin(), d_dest.begin(), less_than_five<T>(), T{0});

  ASSERT_ALMOST_EQUAL(h_data, d_data);
  ASSERT_ALMOST_EQUAL(h_dest, d_dest);
}
DECLARE_VARIABLE_UNITTEST(test_replace_copy_if_stencil);

template <typename T>
THRUST_DISABLE_BROKEN_GCC_VECTORIZER void test_replace_copy_if_stencil_to_discard_iterator(const size_t n)
{
  thrust::host_vector<T> h_data   = unittest::random_samples<T>(n);
  thrust::device_vector<T> d_data = h_data;

  thrust::host_vector<T> h_stencil   = unittest::random_samples<T>(n);
  thrust::device_vector<T> d_stencil = h_stencil;

  const thrust::discard_iterator<> h_result = thrust::replace_copy_if(
    h_data.begin(), h_data.end(), h_stencil.begin(), thrust::make_discard_iterator(), less_than_five<T>(), T{0});

  const thrust::discard_iterator<> d_result = thrust::replace_copy_if(
    d_data.begin(), d_data.end(), d_stencil.begin(), thrust::make_discard_iterator(), less_than_five<T>(), T{0});

  const thrust::discard_iterator<> reference(static_cast<std::ptrdiff_t>(n));

  REQUIRE(reference == h_result);
  REQUIRE(reference == d_result);
}
DECLARE_VARIABLE_UNITTEST(test_replace_copy_if_stencil_to_discard_iterator);
