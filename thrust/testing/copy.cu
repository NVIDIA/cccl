#include <thrust/detail/config.h>

#include <thrust/copy.h>
#include <thrust/device_free.h>
#include <thrust/device_malloc.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/retag.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>

#include <algorithm>
#include <array>
#include <iterator>
#include <list>

#include <unittest/unittest.h>

#if _CCCL_COMPILER(GCC, >=, 11)
#  define THRUST_DISABLE_BROKEN_GCC_VECTORIZER __attribute__((optimize("no-tree-vectorize")))
#else
#  define THRUST_DISABLE_BROKEN_GCC_VECTORIZER
#endif

void TestCopyFromConstIterator()
{
  using T = int;

  std::vector<T> v{0, 1, 2, 3, 4};

  std::vector<int>::const_iterator begin = v.begin();
  std::vector<int>::const_iterator end   = v.end();

  // copy to host_vector
  thrust::host_vector<T> h(5, (T) 10);
  thrust::host_vector<T>::iterator h_result = thrust::copy(begin, end, h.begin());

  thrust::host_vector<T> href{0, 1, 2, 3, 4};
  ASSERT_EQUAL(h, href);
  ASSERT_EQUAL_QUIET(h_result, h.end());

  // copy to device_vector
  thrust::device_vector<T> d(5, (T) 10);
  thrust::device_vector<T>::iterator d_result = thrust::copy(begin, end, d.begin());
  thrust::device_vector<T> dref{0, 1, 2, 3, 4};
  ASSERT_EQUAL(d, dref);
  ASSERT_EQUAL_QUIET(d_result, d.end());
}
DECLARE_UNITTEST(TestCopyFromConstIterator);

void TestCopyToDiscardIterator()
{
  using T = int;

  thrust::host_vector<T> h_input(5, 1);
  thrust::device_vector<T> d_input = h_input;

  thrust::discard_iterator<> reference(5);

  // copy from host_vector
  thrust::discard_iterator<> h_result = thrust::copy(h_input.begin(), h_input.end(), thrust::make_discard_iterator());

  // copy from device_vector
  thrust::discard_iterator<> d_result = thrust::copy(d_input.begin(), d_input.end(), thrust::make_discard_iterator());

  ASSERT_EQUAL_QUIET(reference, h_result);
  ASSERT_EQUAL_QUIET(reference, d_result);
}
DECLARE_UNITTEST(TestCopyToDiscardIterator);

void TestCopyToDiscardIteratorZipped()
{
  using T = int;

  thrust::host_vector<T> h_input(5, 1);
  thrust::device_vector<T> d_input = h_input;

  thrust::host_vector<T> h_output(5);
  thrust::device_vector<T> d_output(5);
  thrust::discard_iterator<> reference(5);

  using Tuple1 = thrust::tuple<thrust::discard_iterator<>, thrust::host_vector<T>::iterator>;
  using Tuple2 = thrust::tuple<thrust::discard_iterator<>, thrust::device_vector<T>::iterator>;

  using ZipIterator1 = thrust::zip_iterator<Tuple1>;
  using ZipIterator2 = thrust::zip_iterator<Tuple2>;

  // copy from host_vector
  ZipIterator1 h_result = thrust::copy(
    thrust::make_zip_iterator(h_input.begin(), h_input.begin()),
    thrust::make_zip_iterator(h_input.end(), h_input.end()),
    thrust::make_zip_iterator(thrust::make_discard_iterator(), h_output.begin()));

  // copy from device_vector
  ZipIterator2 d_result = thrust::copy(
    thrust::make_zip_iterator(d_input.begin(), d_input.begin()),
    thrust::make_zip_iterator(d_input.end(), d_input.end()),
    thrust::make_zip_iterator(thrust::make_discard_iterator(), d_output.begin()));

  ASSERT_EQUAL(h_output, h_input);
  ASSERT_EQUAL(d_output, d_input);
  ASSERT_EQUAL_QUIET(reference, thrust::get<0>(h_result.get_iterator_tuple()));
  ASSERT_EQUAL_QUIET(reference, thrust::get<0>(d_result.get_iterator_tuple()));
}
DECLARE_UNITTEST(TestCopyToDiscardIteratorZipped);

template <class Vector>
void TestCopyMatchingTypes()
{
  using T = typename Vector::value_type;

  Vector v{0, 1, 2, 3, 4};

  // copy to host_vector
  thrust::host_vector<T> h(5, (T) 10);
  typename thrust::host_vector<T>::iterator h_result = thrust::copy(v.begin(), v.end(), h.begin());
  thrust::host_vector<T> href{0, 1, 2, 3, 4};
  ASSERT_EQUAL(h, href);
  ASSERT_EQUAL_QUIET(h_result, h.end());

  // copy to device_vector
  thrust::device_vector<T> d(5, (T) 10);
  typename thrust::device_vector<T>::iterator d_result = thrust::copy(v.begin(), v.end(), d.begin());

  thrust::device_vector<T> dref{0, 1, 2, 3, 4};
  ASSERT_EQUAL(d, dref);
  ASSERT_EQUAL_QUIET(d_result, d.end());
}
DECLARE_VECTOR_UNITTEST(TestCopyMatchingTypes);

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_MSVC(4244) // '=': conversion from 'int' to '_Ty', possible loss of data

template <class Vector>
void TestCopyMixedTypes()
{
  Vector v{0, 1, 2, 3, 4};

  // copy to host_vector with different type
  thrust::host_vector<float> h(5, (float) 10);
  typename thrust::host_vector<float>::iterator h_result = thrust::copy(v.begin(), v.end(), h.begin());
  thrust::host_vector<float> href{0, 1, 2, 3, 4};
  ASSERT_EQUAL(h, href);
  ASSERT_EQUAL_QUIET(h_result, h.end());

  // copy to device_vector with different type
  thrust::device_vector<float> d(5, (float) 10);
  typename thrust::device_vector<float>::iterator d_result = thrust::copy(v.begin(), v.end(), d.begin());
  thrust::device_vector<float> dref{0, 1, 2, 3, 4};
  ASSERT_EQUAL(d, dref);
  ASSERT_EQUAL_QUIET(d_result, d.end());
}
DECLARE_INTEGRAL_VECTOR_UNITTEST(TestCopyMixedTypes);

_CCCL_DIAG_POP

void TestCopyVectorBool()
{
  std::vector<bool> v{true, false, true};

  thrust::host_vector<bool> h(3);
  thrust::device_vector<bool> d(3);

  thrust::copy(v.begin(), v.end(), h.begin());
  thrust::copy(v.begin(), v.end(), d.begin());

  thrust::host_vector<bool> href{true, false, true};
  ASSERT_EQUAL(h, href);

  thrust::device_vector<bool> dref{true, false, true};
  ASSERT_EQUAL(d, dref);
}
DECLARE_UNITTEST(TestCopyVectorBool);

template <class Vector>
void TestCopyListTo()
{
  using T = typename Vector::value_type;

  // copy from list to Vector
  std::list<T> l{0, 1, 2, 3, 4};

  Vector v(l.size());

  typename Vector::iterator v_result = thrust::copy(l.begin(), l.end(), v.begin());

  Vector ref{0, 1, 2, 3, 4};
  ASSERT_EQUAL(v, ref);
  ASSERT_EQUAL_QUIET(v_result, v.end());

  l.clear();

  thrust::copy(v.begin(), v.end(), std::back_insert_iterator<std::list<T>>(l));

  ASSERT_EQUAL(l.size(), 5lu);

  typename std::list<T>::const_iterator iter = l.begin();
  ASSERT_EQUAL(*iter, T(0));
  iter++;
  ASSERT_EQUAL(*iter, T(1));
  iter++;
  ASSERT_EQUAL(*iter, T(2));
  iter++;
  ASSERT_EQUAL(*iter, T(3));
  iter++;
  ASSERT_EQUAL(*iter, T(4));
  iter++;
}
DECLARE_VECTOR_UNITTEST(TestCopyListTo);

template <typename T>
struct is_even
{
  _CCCL_HOST_DEVICE bool operator()(T x)
  {
    return (x & 1) == 0;
  }
};

template <typename T>
struct is_true
{
  _CCCL_HOST_DEVICE bool operator()(T x)
  {
    return x ? true : false;
  }
};

template <typename T>
struct mod_3
{
  _CCCL_HOST_DEVICE unsigned int operator()(T x)
  {
    return x % 3;
  }
};

template <class Vector>
void TestCopyIfSimple()
{
  using T = typename Vector::value_type;

  Vector v{0, 1, 2, 3, 4};

  Vector dest(4);

  typename Vector::iterator dest_end = thrust::copy_if(v.begin(), v.end(), dest.begin(), is_true<T>());

  Vector ref{1, 2, 3, 4};
  ASSERT_EQUAL(ref, dest);
  ASSERT_EQUAL_QUIET(dest.end(), dest_end);
}
DECLARE_VECTOR_UNITTEST(TestCopyIfSimple);

template <typename T>
void TestCopyIf(const size_t n)
{
  thrust::host_vector<T> h_data   = unittest::random_integers<T>(n);
  thrust::device_vector<T> d_data = h_data;

  typename thrust::host_vector<T>::iterator h_new_end;
  typename thrust::device_vector<T>::iterator d_new_end;

  {
    thrust::host_vector<T> h_result(n);
    thrust::device_vector<T> d_result(n);

    h_new_end = thrust::copy_if(h_data.begin(), h_data.end(), h_result.begin(), is_true<T>());
    d_new_end = thrust::copy_if(d_data.begin(), d_data.end(), d_result.begin(), is_true<T>());

    h_result.resize(h_new_end - h_result.begin());
    d_result.resize(d_new_end - d_result.begin());

    ASSERT_EQUAL(h_result, d_result);
  }
}
DECLARE_INTEGRAL_VARIABLE_UNITTEST(TestCopyIf);

template <typename T>
void TestCopyIfIntegral(const size_t n)
{
  thrust::host_vector<T> h_data   = unittest::random_integers<T>(n);
  thrust::device_vector<T> d_data = h_data;

  typename thrust::host_vector<T>::iterator h_new_end;
  typename thrust::device_vector<T>::iterator d_new_end;

  // test with Predicate that returns a bool
  {
    thrust::host_vector<T> h_result(n);
    thrust::device_vector<T> d_result(n);

    h_new_end = thrust::copy_if(h_data.begin(), h_data.end(), h_result.begin(), is_even<T>());
    d_new_end = thrust::copy_if(d_data.begin(), d_data.end(), d_result.begin(), is_even<T>());

    h_result.resize(h_new_end - h_result.begin());
    d_result.resize(d_new_end - d_result.begin());

    ASSERT_EQUAL(h_result, d_result);
  }

  // test with Predicate that returns a non-bool
  {
    thrust::host_vector<T> h_result(n);
    thrust::device_vector<T> d_result(n);

    h_new_end = thrust::copy_if(h_data.begin(), h_data.end(), h_result.begin(), mod_3<T>());
    d_new_end = thrust::copy_if(d_data.begin(), d_data.end(), d_result.begin(), mod_3<T>());

    h_result.resize(h_new_end - h_result.begin());
    d_result.resize(d_new_end - d_result.begin());

    ASSERT_EQUAL(h_result, d_result);
  }
}
DECLARE_INTEGRAL_VARIABLE_UNITTEST(TestCopyIfIntegral);

template <typename T>
void TestCopyIfSequence(const size_t n)
{
  thrust::host_vector<T> h_data(n);
  thrust::sequence(h_data.begin(), h_data.end());
  thrust::device_vector<T> d_data(n);
  thrust::sequence(d_data.begin(), d_data.end());

  typename thrust::host_vector<T>::iterator h_new_end;
  typename thrust::device_vector<T>::iterator d_new_end;

  // test with Predicate that returns a bool
  {
    thrust::host_vector<T> h_result(n);
    thrust::device_vector<T> d_result(n);

    h_new_end = thrust::copy_if(h_data.begin(), h_data.end(), h_result.begin(), is_even<T>());
    d_new_end = thrust::copy_if(d_data.begin(), d_data.end(), d_result.begin(), is_even<T>());

    h_result.resize(h_new_end - h_result.begin());
    d_result.resize(d_new_end - d_result.begin());

    ASSERT_EQUAL(h_result, d_result);
  }

  // test with Predicate that returns a non-bool
  {
    thrust::host_vector<T> h_result(n);
    thrust::device_vector<T> d_result(n);

    h_new_end = thrust::copy_if(h_data.begin(), h_data.end(), h_result.begin(), mod_3<T>());
    d_new_end = thrust::copy_if(d_data.begin(), d_data.end(), d_result.begin(), mod_3<T>());

    h_result.resize(h_new_end - h_result.begin());
    d_result.resize(d_new_end - d_result.begin());

    ASSERT_EQUAL(h_result, d_result);
  }
}
DECLARE_INTEGRAL_VARIABLE_UNITTEST(TestCopyIfSequence);

template <class Vector>
void TestCopyIfStencilSimple()
{
  using T = typename Vector::value_type;

  Vector v{0, 1, 2, 3, 4};
  Vector s{1, 1, 0, 1, 0};

  Vector dest(3);

  typename Vector::iterator dest_end = thrust::copy_if(v.begin(), v.end(), s.begin(), dest.begin(), is_true<T>());

  Vector ref{0, 1, 3};
  ASSERT_EQUAL(ref, dest);
  ASSERT_EQUAL_QUIET(dest.end(), dest_end);
}
DECLARE_VECTOR_UNITTEST(TestCopyIfStencilSimple);

template <typename T>
void TestCopyIfStencil(const size_t n)
{
  thrust::host_vector<T> h_data(n);
  thrust::sequence(h_data.begin(), h_data.end());
  thrust::device_vector<T> d_data(n);
  thrust::sequence(d_data.begin(), d_data.end());

  thrust::host_vector<T> h_stencil   = unittest::random_integers<T>(n);
  thrust::device_vector<T> d_stencil = unittest::random_integers<T>(n);

  typename thrust::host_vector<T>::iterator h_new_end;
  typename thrust::device_vector<T>::iterator d_new_end;

  {
    thrust::host_vector<T> h_result(n);
    thrust::device_vector<T> d_result(n);

    h_new_end = thrust::copy_if(h_data.begin(), h_data.end(), h_stencil.begin(), h_result.begin(), is_even<T>());
    d_new_end = thrust::copy_if(d_data.begin(), d_data.end(), d_stencil.begin(), d_result.begin(), is_even<T>());

    h_result.resize(h_new_end - h_result.begin());
    d_result.resize(d_new_end - d_result.begin());

    ASSERT_EQUAL(h_result, d_result);
  }
}
DECLARE_INTEGRAL_VARIABLE_UNITTEST(TestCopyIfStencil);

namespace
{

struct object_with_non_trivial_ctor
{
  // This struct will only properly assign if its `magic` member is
  // set to this certain number.
  static constexpr int MAGIC = 923390;

  int field;
  int magic;

  _CCCL_HOST_DEVICE object_with_non_trivial_ctor()
  {
    magic = MAGIC;
    field = 0;
  }
  _CCCL_HOST_DEVICE object_with_non_trivial_ctor(int f)
  {
    magic = MAGIC;
    field = f;
  }

  object_with_non_trivial_ctor(const object_with_non_trivial_ctor& x) = default;

  // This non-trivial assignment requires that `this` points to initialized
  // memory
  _CCCL_HOST_DEVICE object_with_non_trivial_ctor& operator=(const object_with_non_trivial_ctor& x)
  {
    // To really copy over x's field value, require we have magic value set.
    // If copy_if copies to uninitialized bits, the field will rarely be 923390.
    if (magic == MAGIC)
    {
      field = x.field;
    }
    return *this;
  }
};

struct always_true
{
  _CCCL_HOST_DEVICE bool operator()(const object_with_non_trivial_ctor&)
  {
    return true;
  }
};

} // namespace

void TestCopyIfNonTrivial()
{
  // Attempting to copy an object_with_non_trivial_ctor into uninitialized
  // memory will fail:
  {
    static constexpr size_t BufferAlign = alignof(object_with_non_trivial_ctor);
    static constexpr size_t BufferSize  = sizeof(object_with_non_trivial_ctor);
    alignas(BufferAlign) std::array<unsigned char, BufferSize> buffer;

    // Fill buffer with 0s to prevent warnings about uninitialized reads while
    // ensure that the 'magic number' mechanism works as intended:
    std::fill(buffer.begin(), buffer.end(), static_cast<unsigned char>(0));

    object_with_non_trivial_ctor initialized;
    object_with_non_trivial_ctor* uninitialized = reinterpret_cast<object_with_non_trivial_ctor*>(buffer.data());

    object_with_non_trivial_ctor source(42);
    initialized    = source;
    *uninitialized = source;

    ASSERT_EQUAL(42, initialized.field);
    ASSERT_NOT_EQUAL(42, uninitialized->field);
  }

  // This test ensures that we use placement new instead of assigning
  // to uninitialized memory. See Thrust Github issue #1153.
  thrust::device_vector<object_with_non_trivial_ctor> a(10, object_with_non_trivial_ctor(99));
  thrust::device_vector<object_with_non_trivial_ctor> b(10);

  thrust::copy_if(a.begin(), a.end(), b.begin(), always_true());

  for (int i = 0; i < 10; i++)
  {
    object_with_non_trivial_ctor ha(a[i]);
    object_with_non_trivial_ctor hb(b[i]);
    int ia = ha.field;
    int ib = hb.field;

    ASSERT_EQUAL(ia, ib);
  }
}
DECLARE_UNITTEST(TestCopyIfNonTrivial);

template <typename Vector>
void TestCopyCountingIterator()
{
  using T = typename Vector::value_type;

  thrust::counting_iterator<T> iter(1);

  Vector vec(4);

  thrust::copy(iter, iter + 4, vec.begin());

  ASSERT_EQUAL(vec[0], 1);
  ASSERT_EQUAL(vec[1], 2);
  ASSERT_EQUAL(vec[2], 3);
  ASSERT_EQUAL(vec[3], 4);
}
DECLARE_INTEGRAL_VECTOR_UNITTEST(TestCopyCountingIterator);

template <typename Vector>
THRUST_DISABLE_BROKEN_GCC_VECTORIZER void TestCopyZipIterator()
{
  using T = typename Vector::value_type;

  // initializer list doesn't work with GCC when
  // Vector = thrust::host_vector<signed char>
  // Vector v1{1, 2, 3};

  Vector v1(3);
  v1[0] = 1;
  v1[1] = 2;
  v1[2] = 3;
  Vector v2(3);
  v2[0] = 4;
  v2[1] = 5;
  v2[2] = 6;
  Vector v3(3, T(0));
  Vector v4(3, T(0));

  thrust::copy(thrust::make_zip_iterator(v1.begin(), v2.begin()),
               thrust::make_zip_iterator(v1.end(), v2.end()),
               thrust::make_zip_iterator(v3.begin(), v4.begin()));

  ASSERT_EQUAL(v1, v3);
  ASSERT_EQUAL(v2, v4);
};
DECLARE_VECTOR_UNITTEST(TestCopyZipIterator);

template <typename Vector>
void TestCopyConstantIteratorToZipIterator()
{
  using T = typename Vector::value_type;

  Vector v1(3, T(0));
  Vector v2(3, T(0));

  thrust::copy(thrust::make_constant_iterator(thrust::tuple<T, T>(4, 7)),
               thrust::make_constant_iterator(thrust::tuple<T, T>(4, 7)) + v1.size(),
               thrust::make_zip_iterator(v1.begin(), v2.begin()));

  Vector ref1{4, 4, 4};
  Vector ref2{7, 7, 7};
  ASSERT_EQUAL(v1, ref1);
  ASSERT_EQUAL(v2, ref2);
};
DECLARE_VECTOR_UNITTEST(TestCopyConstantIteratorToZipIterator);

template <typename InputIterator, typename OutputIterator>
OutputIterator copy(my_system& system, InputIterator, InputIterator, OutputIterator result)
{
  system.validate_dispatch();
  return result;
}

void TestCopyDispatchExplicit()
{
  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::copy(sys, vec.begin(), vec.end(), vec.begin());

  ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestCopyDispatchExplicit);

template <typename InputIterator, typename OutputIterator>
OutputIterator copy(my_tag, InputIterator, InputIterator, OutputIterator result)
{
  *result = 13;
  return result;
}

void TestCopyDispatchImplicit()
{
  thrust::device_vector<int> vec(1);

  thrust::copy(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.end()), thrust::retag<my_tag>(vec.begin()));

  ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestCopyDispatchImplicit);

template <typename InputIterator, typename OutputIterator, typename Predicate>
OutputIterator copy_if(my_system& system, InputIterator, InputIterator, OutputIterator result, Predicate)
{
  system.validate_dispatch();
  return result;
}

void TestCopyIfDispatchExplicit()
{
  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::copy_if(sys, vec.begin(), vec.end(), vec.begin(), 0);

  ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestCopyIfDispatchExplicit);

template <typename InputIterator, typename OutputIterator, typename Predicate>
OutputIterator copy_if(my_tag, InputIterator, InputIterator, OutputIterator result, Predicate)
{
  *result = 13;
  return result;
}

void TestCopyIfDispatchImplicit()
{
  thrust::device_vector<int> vec(1);

  thrust::copy_if(
    thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.end()), thrust::retag<my_tag>(vec.begin()), 0);

  ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestCopyIfDispatchImplicit);

template <typename InputIterator1, typename InputIterator2, typename OutputIterator, typename Predicate>
OutputIterator
copy_if(my_system& system, InputIterator1, InputIterator1, InputIterator2, OutputIterator result, Predicate)
{
  system.validate_dispatch();
  return result;
}

void TestCopyIfStencilDispatchExplicit()
{
  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::copy_if(sys, vec.begin(), vec.end(), vec.begin(), vec.begin(), 0);

  ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestCopyIfStencilDispatchExplicit);

template <typename InputIterator1, typename InputIterator2, typename OutputIterator, typename Predicate>
OutputIterator copy_if(my_tag, InputIterator1, InputIterator1, InputIterator2, OutputIterator result, Predicate)
{
  *result = 13;
  return result;
}

void TestCopyIfStencilDispatchImplicit()
{
  thrust::device_vector<int> vec(1);

  thrust::copy_if(
    thrust::retag<my_tag>(vec.begin()),
    thrust::retag<my_tag>(vec.end()),
    thrust::retag<my_tag>(vec.begin()),
    thrust::retag<my_tag>(vec.begin()),
    0);

  ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestCopyIfStencilDispatchImplicit);

#ifndef THRUST_FORCE_32_BIT_OFFSET_TYPE

struct only_set_when_expected_it
{
  long long expected;
  bool* flag;

  _CCCL_HOST_DEVICE only_set_when_expected_it operator++() const
  {
    return *this;
  }
  _CCCL_HOST_DEVICE only_set_when_expected_it operator*() const
  {
    return *this;
  }
  template <typename Difference>
  _CCCL_HOST_DEVICE only_set_when_expected_it operator+(Difference) const
  {
    return *this;
  }
  template <typename Difference>
  _CCCL_HOST_DEVICE only_set_when_expected_it operator+=(Difference) const
  {
    return *this;
  }
  template <typename Index>
  _CCCL_HOST_DEVICE only_set_when_expected_it operator[](Index) const
  {
    return *this;
  }

  _CCCL_DEVICE void operator=(long long value) const
  {
    if (value == expected)
    {
      *flag = true;
    }
  }
};

THRUST_NAMESPACE_BEGIN
namespace detail
{
// We need this type to pass as a non-const ref for unary_transform_functor
// to compile:
template <>
inline constexpr bool is_non_const_reference_v<only_set_when_expected_it> = true;
} // end namespace detail
THRUST_NAMESPACE_END

namespace std
{
template <>
struct iterator_traits<only_set_when_expected_it>
{
  using value_type        = long long;
  using reference         = only_set_when_expected_it;
  using iterator_category = thrust::random_access_device_iterator_tag;
  using difference_type   = ::cuda::std::ptrdiff_t;
};
} // namespace std

_CCCL_BEGIN_NAMESPACE_CUDA_STD
template <>
struct iterator_traits<only_set_when_expected_it>
{
  using value_type        = long long;
  using reference         = only_set_when_expected_it;
  using iterator_category = thrust::random_access_device_iterator_tag;
  using difference_type   = ::cuda::std::ptrdiff_t;
};
_CCCL_END_NAMESPACE_CUDA_STD

void TestCopyWithBigIndexesHelper(int magnitude)
{
  thrust::counting_iterator<long long> begin(0);
  thrust::counting_iterator<long long> end = begin + (1ll << magnitude);
  ASSERT_EQUAL(::cuda::std::distance(begin, end), 1ll << magnitude);

  thrust::device_ptr<bool> has_executed = thrust::device_malloc<bool>(1);
  *has_executed                         = false;

  only_set_when_expected_it out = {(1ll << magnitude) - 1, thrust::raw_pointer_cast(has_executed)};

  thrust::copy(thrust::device, begin, end, out);

  bool has_executed_h = *has_executed;
  thrust::device_free(has_executed);

  ASSERT_EQUAL(has_executed_h, true);
}

void TestCopyWithBigIndexes()
{
  TestCopyWithBigIndexesHelper(30);
  TestCopyWithBigIndexesHelper(31);
  TestCopyWithBigIndexesHelper(32);
  TestCopyWithBigIndexesHelper(33);
}
DECLARE_UNITTEST(TestCopyWithBigIndexes);

#endif
