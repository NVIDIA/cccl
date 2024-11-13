#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/retag.h>
#include <thrust/replace.h>

#include <unittest/unittest.h>

// There is a unfortunate miscompilation of the gcc-11 vectorizer leading to OOB writes
// Adding this attribute suffices that this miscompilation does not appear anymore
#if defined(_CCCL_COMPILER_GCC) && __GNUC__ >= 11
#  define THRUST_DISABLE_BROKEN_GCC_VECTORIZER __attribute__((optimize("no-tree-vectorize")))
#else
#  define THRUST_DISABLE_BROKEN_GCC_VECTORIZER
#endif

// GCC 12 + omp + c++11 miscompiles some test cases and emits spurious warnings.
#if defined(_CCCL_COMPILER_GCC) && __GNUC__ == 12 && THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_OMP \
  && _CCCL_STD_VER == 2011
#  define THRUST_GCC12_OMP_MISCOMPILE
#endif

// New GCC, new miscompile. 13 + TBB this time.
#if defined(_CCCL_COMPILER_GCC) && __GNUC__ == 13 && THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_TBB
#  define THRUST_GCC13_TBB_MISCOMPILE
#endif

template <class Vector>
void TestReplaceSimple()
{
  using T = typename Vector::value_type;

  Vector data{1, 2, 1, 3, 2};

  thrust::replace(data.begin(), data.end(), (T) 1, (T) 4);
  thrust::replace(data.begin(), data.end(), (T) 2, (T) 5);

  Vector result{4, 5, 4, 3, 5};

  ASSERT_EQUAL(data, result);
}
DECLARE_VECTOR_UNITTEST(TestReplaceSimple);

template <typename ForwardIterator, typename T>
void replace(my_system& system, ForwardIterator, ForwardIterator, const T&, const T&)
{
  system.validate_dispatch();
}

void TestReplaceDispatchExplicit()
{
  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::replace(sys, vec.begin(), vec.begin(), 0, 0);

  ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestReplaceDispatchExplicit);

template <typename ForwardIterator, typename T>
void replace(my_tag, ForwardIterator first, ForwardIterator, const T&, const T&)
{
  *first = 13;
}

void TestReplaceDispatchImplicit()
{
  thrust::device_vector<int> vec(1);

  thrust::replace(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()), 0, 0);

  ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestReplaceDispatchImplicit);

template <typename T>
void TestReplace(const size_t n)
{
  thrust::host_vector<T> h_data   = unittest::random_samples<T>(n);
  thrust::device_vector<T> d_data = h_data;

  T old_value = 0;
  T new_value = 1;

  thrust::replace(h_data.begin(), h_data.end(), old_value, new_value);
  thrust::replace(d_data.begin(), d_data.end(), old_value, new_value);

  ASSERT_ALMOST_EQUAL(h_data, d_data);
}
DECLARE_VARIABLE_UNITTEST(TestReplace);

#ifndef THRUST_GCC13_TBB_MISCOMPILE
#  ifndef THRUST_GCC12_OMP_MISCOMPILE
template <class Vector>
THRUST_DISABLE_BROKEN_GCC_VECTORIZER void TestReplaceCopySimple()
{
  using T = typename Vector::value_type;

  Vector data{1, 2, 1, 3, 2};

  Vector dest(5);

  thrust::replace_copy(data.begin(), data.end(), dest.begin(), (T) 1, (T) 4);
  thrust::replace_copy(dest.begin(), dest.end(), dest.begin(), (T) 2, (T) 5);

  Vector result{4, 5, 4, 3, 5};
  ASSERT_EQUAL(dest, result);
}
DECLARE_VECTOR_UNITTEST(TestReplaceCopySimple);
#  endif
#endif

template <typename InputIterator, typename OutputIterator, typename T>
OutputIterator replace_copy(my_system& system, InputIterator, InputIterator, OutputIterator result, const T&, const T&)
{
  system.validate_dispatch();
  return result;
}

void TestReplaceCopyDispatchExplicit()
{
  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::replace_copy(sys, vec.begin(), vec.begin(), vec.begin(), 0, 0);

  ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestReplaceCopyDispatchExplicit);

template <typename InputIterator, typename OutputIterator, typename T>
OutputIterator replace_copy(my_tag, InputIterator, InputIterator, OutputIterator result, const T&, const T&)
{
  *result = 13;
  return result;
}

void TestReplaceCopyDispatchImplicit()
{
  thrust::device_vector<int> vec(1);

  thrust::replace_copy(
    thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()), 0, 0);

  ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestReplaceCopyDispatchImplicit);

template <typename T>
void TestReplaceCopy(const size_t n)
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
DECLARE_VARIABLE_UNITTEST(TestReplaceCopy);

template <typename T>
void TestReplaceCopyToDiscardIterator(const size_t n)
{
  thrust::host_vector<T> h_data   = unittest::random_samples<T>(n);
  thrust::device_vector<T> d_data = h_data;

  T old_value = 0;
  T new_value = 1;

  thrust::discard_iterator<> h_result =
    thrust::replace_copy(h_data.begin(), h_data.end(), thrust::make_discard_iterator(), old_value, new_value);

  thrust::discard_iterator<> d_result =
    thrust::replace_copy(d_data.begin(), d_data.end(), thrust::make_discard_iterator(), old_value, new_value);

  thrust::discard_iterator<> reference(n);

  ASSERT_EQUAL_QUIET(reference, h_result);
  ASSERT_EQUAL_QUIET(reference, d_result);
}
DECLARE_VARIABLE_UNITTEST(TestReplaceCopyToDiscardIterator);

template <typename T>
struct less_than_five
{
  _CCCL_HOST_DEVICE bool operator()(const T& val) const
  {
    return val < 5;
  }
};

template <class Vector>
void TestReplaceIfSimple()
{
  using T = typename Vector::value_type;

  Vector data{1, 3, 4, 6, 5};

  thrust::replace_if(data.begin(), data.end(), less_than_five<T>(), (T) 0);

  Vector result{0, 0, 0, 6, 5};

  ASSERT_EQUAL(data, result);
}
DECLARE_VECTOR_UNITTEST(TestReplaceIfSimple);

template <typename ForwardIterator, typename Predicate, typename T>
void replace_if(my_system& system, ForwardIterator, ForwardIterator, Predicate, const T&)
{
  system.validate_dispatch();
}

void TestReplaceIfDispatchExplicit()
{
  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::replace_if(sys, vec.begin(), vec.begin(), 0, 0);

  ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestReplaceIfDispatchExplicit);

template <typename ForwardIterator, typename Predicate, typename T>
void replace_if(my_tag, ForwardIterator first, ForwardIterator, Predicate, const T&)
{
  *first = 13;
}

void TestReplaceIfDispatchImplicit()
{
  thrust::device_vector<int> vec(1);

  thrust::replace_if(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()), 0, 0);

  ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestReplaceIfDispatchImplicit);

template <class Vector>
THRUST_DISABLE_BROKEN_GCC_VECTORIZER void TestReplaceIfStencilSimple()
{
  using T = typename Vector::value_type;

  Vector data{1, 3, 4, 6, 5};

  Vector stencil{5, 4, 6, 3, 7};
  thrust::replace_if(data.begin(), data.end(), stencil.begin(), less_than_five<T>(), (T) 0);

  Vector result{1, 0, 4, 0, 5};

  ASSERT_EQUAL(data, result);
}
DECLARE_VECTOR_UNITTEST(TestReplaceIfStencilSimple);

template <typename ForwardIterator, typename InputIterator, typename Predicate, typename T>
void replace_if(my_system& system, ForwardIterator, ForwardIterator, InputIterator, Predicate, const T&)
{
  system.validate_dispatch();
}

void TestReplaceIfStencilDispatchExplicit()
{
  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::replace_if(sys, vec.begin(), vec.begin(), vec.begin(), 0, 0);

  ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestReplaceIfStencilDispatchExplicit);

template <typename ForwardIterator, typename InputIterator, typename Predicate, typename T>
void replace_if(my_tag, ForwardIterator first, ForwardIterator, InputIterator, Predicate, const T&)
{
  *first = 13;
}

void TestReplaceIfStencilDispatchImplicit()
{
  thrust::device_vector<int> vec(1);

  thrust::replace_if(
    thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()), 0, 0);

  ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestReplaceIfStencilDispatchImplicit);

template <typename T>
THRUST_DISABLE_BROKEN_GCC_VECTORIZER void TestReplaceIf(const size_t n)
{
  thrust::host_vector<T> h_data   = unittest::random_samples<T>(n);
  thrust::device_vector<T> d_data = h_data;

  thrust::replace_if(h_data.begin(), h_data.end(), less_than_five<T>(), (T) 0);
  thrust::replace_if(d_data.begin(), d_data.end(), less_than_five<T>(), (T) 0);

  ASSERT_ALMOST_EQUAL(h_data, d_data);
}
DECLARE_VARIABLE_UNITTEST(TestReplaceIf);

template <typename T>
THRUST_DISABLE_BROKEN_GCC_VECTORIZER void TestReplaceIfStencil(const size_t n)
{
  thrust::host_vector<T> h_data   = unittest::random_samples<T>(n);
  thrust::device_vector<T> d_data = h_data;

  thrust::host_vector<T> h_stencil   = unittest::random_samples<T>(n);
  thrust::device_vector<T> d_stencil = h_stencil;

  thrust::replace_if(h_data.begin(), h_data.end(), h_stencil.begin(), less_than_five<T>(), (T) 0);
  thrust::replace_if(d_data.begin(), d_data.end(), d_stencil.begin(), less_than_five<T>(), (T) 0);

  ASSERT_ALMOST_EQUAL(h_data, d_data);
}
DECLARE_VARIABLE_UNITTEST(TestReplaceIfStencil);

template <class Vector>
THRUST_DISABLE_BROKEN_GCC_VECTORIZER void TestReplaceCopyIfSimple()
{
  using T = typename Vector::value_type;

  Vector data{1, 3, 4, 6, 5};

  Vector dest(5);

  thrust::replace_copy_if(data.begin(), data.end(), dest.begin(), less_than_five<T>(), (T) 0);

  Vector result{0, 0, 0, 6, 5};
  ASSERT_EQUAL(dest, result);
}
DECLARE_VECTOR_UNITTEST(TestReplaceCopyIfSimple);

template <typename InputIterator, typename OutputIterator, typename Predicate, typename T>
OutputIterator
replace_copy_if(my_system& system, InputIterator, InputIterator, OutputIterator result, Predicate, const T&)
{
  system.validate_dispatch();
  return result;
}

void TestReplaceCopyIfDispatchExplicit()
{
  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::replace_copy_if(sys, vec.begin(), vec.begin(), vec.begin(), 0, 0);

  ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestReplaceCopyIfDispatchExplicit);

template <typename InputIterator, typename OutputIterator, typename Predicate, typename T>
OutputIterator replace_copy_if(my_tag, InputIterator, InputIterator, OutputIterator result, Predicate, const T&)
{
  *result = 13;
  return result;
}

void TestReplaceCopyIfDispatchImplicit()
{
  thrust::device_vector<int> vec(1);

  thrust::replace_copy_if(
    thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()), 0, 0);

  ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestReplaceCopyIfDispatchImplicit);

template <class Vector>
THRUST_DISABLE_BROKEN_GCC_VECTORIZER void TestReplaceCopyIfStencilSimple()
{
  using T = typename Vector::value_type;

  Vector data{1, 3, 4, 6, 5};
  Vector stencil{1, 5, 4, 7, 8};

  Vector dest(5);

  thrust::replace_copy_if(data.begin(), data.end(), stencil.begin(), dest.begin(), less_than_five<T>(), (T) 0);

  Vector result{0, 3, 0, 6, 5};

  ASSERT_EQUAL(dest, result);
}
DECLARE_VECTOR_UNITTEST(TestReplaceCopyIfStencilSimple);

template <typename InputIterator1, typename InputIterator2, typename OutputIterator, typename Predicate, typename T>
OutputIterator replace_copy_if(
  my_system& system, InputIterator1, InputIterator1, InputIterator2, OutputIterator result, Predicate, const T&)
{
  system.validate_dispatch();
  return result;
}

void TestReplaceCopyIfStencilDispatchExplicit()
{
  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::replace_copy_if(sys, vec.begin(), vec.begin(), vec.begin(), vec.begin(), 0, 0);

  ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestReplaceCopyIfStencilDispatchExplicit);

template <typename InputIterator1, typename InputIterator2, typename OutputIterator, typename Predicate, typename T>
OutputIterator
replace_copy_if(my_tag, InputIterator1, InputIterator1, InputIterator2, OutputIterator result, Predicate, const T&)
{
  *result = 13;
  return result;
}

void TestReplaceCopyIfStencilDispatchImplicit()
{
  thrust::device_vector<int> vec(1);

  thrust::replace_copy_if(
    thrust::retag<my_tag>(vec.begin()),
    thrust::retag<my_tag>(vec.begin()),
    thrust::retag<my_tag>(vec.begin()),
    thrust::retag<my_tag>(vec.begin()),
    0,
    0);

  ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestReplaceCopyIfStencilDispatchImplicit);

template <typename T>
THRUST_DISABLE_BROKEN_GCC_VECTORIZER void TestReplaceCopyIf(const size_t n)
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
DECLARE_VARIABLE_UNITTEST(TestReplaceCopyIf);

template <typename T>
THRUST_DISABLE_BROKEN_GCC_VECTORIZER void TestReplaceCopyIfToDiscardIterator(const size_t n)
{
  thrust::host_vector<T> h_data   = unittest::random_samples<T>(n);
  thrust::device_vector<T> d_data = h_data;

  thrust::discard_iterator<> h_result =
    thrust::replace_copy_if(h_data.begin(), h_data.end(), thrust::make_discard_iterator(), less_than_five<T>(), T{0});

  thrust::discard_iterator<> d_result =
    thrust::replace_copy_if(d_data.begin(), d_data.end(), thrust::make_discard_iterator(), less_than_five<T>(), T{0});

  thrust::discard_iterator<> reference(n);

  ASSERT_EQUAL_QUIET(reference, h_result);
  ASSERT_EQUAL_QUIET(reference, d_result);
}
DECLARE_VARIABLE_UNITTEST(TestReplaceCopyIfToDiscardIterator);

template <typename T>
THRUST_DISABLE_BROKEN_GCC_VECTORIZER void TestReplaceCopyIfStencil(const size_t n)
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
DECLARE_VARIABLE_UNITTEST(TestReplaceCopyIfStencil);

template <typename T>
THRUST_DISABLE_BROKEN_GCC_VECTORIZER void TestReplaceCopyIfStencilToDiscardIterator(const size_t n)
{
  thrust::host_vector<T> h_data   = unittest::random_samples<T>(n);
  thrust::device_vector<T> d_data = h_data;

  thrust::host_vector<T> h_stencil   = unittest::random_samples<T>(n);
  thrust::device_vector<T> d_stencil = h_stencil;

  thrust::discard_iterator<> h_result = thrust::replace_copy_if(
    h_data.begin(), h_data.end(), h_stencil.begin(), thrust::make_discard_iterator(), less_than_five<T>(), T{0});

  thrust::discard_iterator<> d_result = thrust::replace_copy_if(
    d_data.begin(), d_data.end(), d_stencil.begin(), thrust::make_discard_iterator(), less_than_five<T>(), T{0});

  thrust::discard_iterator<> reference(n);

  ASSERT_EQUAL_QUIET(reference, h_result);
  ASSERT_EQUAL_QUIET(reference, d_result);
}
DECLARE_VARIABLE_UNITTEST(TestReplaceCopyIfStencilToDiscardIterator);
