#include <thrust/copy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/retag.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>

#include <iterator>
#include <list>

#include <unittest/unittest.h>

void TestCopyNFromConstIterator()
{
  using T = int;

  std::vector<T> v{0, 1, 2, 3, 4};

  std::vector<int>::const_iterator begin = v.begin();

  // copy to host_vector
  thrust::host_vector<T> h(5, (T) 10);
  thrust::host_vector<T>::iterator h_result = thrust::copy_n(begin, h.size(), h.begin());

  ASSERT_EQUAL_QUIET(h_result, h.end());

  // copy to device_vector
  thrust::device_vector<T> d(5, (T) 10);
  thrust::device_vector<T>::iterator d_result = thrust::copy_n(begin, d.size(), d.begin());

  thrust::device_vector<T> dref{0, 1, 2, 3, 4};
  ASSERT_EQUAL(d, dref);
  ASSERT_EQUAL_QUIET(d_result, d.end());
}
DECLARE_UNITTEST(TestCopyNFromConstIterator);

void TestCopyNToDiscardIterator()
{
  using T = int;

  thrust::host_vector<T> h_input(5, 1);
  thrust::device_vector<T> d_input = h_input;

  // copy from host_vector
  thrust::discard_iterator<> h_result =
    thrust::copy_n(h_input.begin(), h_input.size(), thrust::make_discard_iterator());

  // copy from device_vector
  thrust::discard_iterator<> d_result =
    thrust::copy_n(d_input.begin(), d_input.size(), thrust::make_discard_iterator());

  thrust::discard_iterator<> reference(5);

  ASSERT_EQUAL_QUIET(reference, h_result);
  ASSERT_EQUAL_QUIET(reference, d_result);
}
DECLARE_UNITTEST(TestCopyNToDiscardIterator);

template <class Vector>
void TestCopyNMatchingTypes()
{
  using T = typename Vector::value_type;

  Vector v{0, 1, 2, 3, 4};

  // copy to host_vector
  thrust::host_vector<T> h(5, (T) 10);
  typename thrust::host_vector<T>::iterator h_result = thrust::copy_n(v.begin(), v.size(), h.begin());
  thrust::host_vector<T> href{0, 1, 2, 3, 4};
  ASSERT_EQUAL(h, href);
  ASSERT_EQUAL_QUIET(h_result, h.end());

  // copy to device_vector
  thrust::device_vector<T> d(5, (T) 10);
  typename thrust::device_vector<T>::iterator d_result = thrust::copy_n(v.begin(), v.size(), d.begin());
  thrust::device_vector<T> dref{0, 1, 2, 3, 4};
  ASSERT_EQUAL(d, dref);
  ASSERT_EQUAL_QUIET(d_result, d.end());
}
DECLARE_VECTOR_UNITTEST(TestCopyNMatchingTypes);

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_MSVC(4244) // '=': conversion from 'int' to '_Ty', possible loss of data

template <class Vector>
void TestCopyNMixedTypes()
{
  Vector v{0, 1, 2, 3, 4};

  // copy to host_vector with different type
  thrust::host_vector<float> h(5, (float) 10);
  typename thrust::host_vector<float>::iterator h_result = thrust::copy_n(v.begin(), v.size(), h.begin());

  thrust::host_vector<float> href{0, 1, 2, 3, 4};
  ASSERT_EQUAL(h, href);
  ASSERT_EQUAL_QUIET(h_result, h.end());

  // copy to device_vector with different type
  thrust::device_vector<float> d(5, (float) 10);
  typename thrust::device_vector<float>::iterator d_result = thrust::copy_n(v.begin(), v.size(), d.begin());
  thrust::device_vector<float> dref{0, 1, 2, 3, 4};
  ASSERT_EQUAL(d, dref);
  ASSERT_EQUAL_QUIET(d_result, d.end());
}
DECLARE_INTEGRAL_VECTOR_UNITTEST(TestCopyNMixedTypes);

_CCCL_DIAG_POP

void TestCopyNVectorBool()
{
  std::vector<bool> v{true, false, true};

  thrust::host_vector<bool> h(3);
  thrust::device_vector<bool> d(3);

  thrust::copy_n(v.begin(), v.size(), h.begin());
  thrust::copy_n(v.begin(), v.size(), d.begin());

  thrust::host_vector<bool> href{true, false, true};
  ASSERT_EQUAL(h, href);

  thrust::device_vector<bool> dref{true, false, true};
  ASSERT_EQUAL(d, dref);

  ASSERT_EQUAL(d, dref);
}
DECLARE_UNITTEST(TestCopyNVectorBool);

template <class Vector>
void TestCopyNListTo()
{
  using T = typename Vector::value_type;

  // copy from list to Vector
  std::list<T> l{0, 1, 2, 3, 4};

  Vector v(l.size());

  typename Vector::iterator v_result = thrust::copy_n(l.begin(), l.size(), v.begin());

  Vector ref{0, 1, 2, 3, 4};
  ASSERT_EQUAL(v, ref);
  ASSERT_EQUAL_QUIET(v_result, v.end());

  l.clear();

  thrust::copy_n(v.begin(), v.size(), std::back_insert_iterator<std::list<T>>(l));

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
DECLARE_VECTOR_UNITTEST(TestCopyNListTo);

template <typename Vector>
void TestCopyNCountingIterator()
{
  using T = typename Vector::value_type;

  thrust::counting_iterator<T> iter(1);

  Vector vec(4);

  thrust::copy_n(iter, 4, vec.begin());

  Vector ref{1, 2, 3, 4};
  ASSERT_EQUAL(vec, ref);
}
DECLARE_INTEGRAL_VECTOR_UNITTEST(TestCopyNCountingIterator);

template <typename Vector>
void TestCopyNZipIterator()
{
  using T = typename Vector::value_type;

  Vector v1{1, 2, 3, 4};
  Vector v2{4, 5, 6, 7};
  Vector v3(4, T(0));
  Vector v4(4, T(0));

  thrust::copy_n(thrust::make_zip_iterator(thrust::make_tuple(v1.begin(), v2.begin())),
                 4,
                 thrust::make_zip_iterator(thrust::make_tuple(v3.begin(), v4.begin())));

  ASSERT_EQUAL(v1, v3);
  ASSERT_EQUAL(v2, v4);
};
DECLARE_VECTOR_UNITTEST(TestCopyNZipIterator);

template <typename Vector>
void TestCopyNConstantIteratorToZipIterator()
{
  using T = typename Vector::value_type;

  Vector v1(4, T(0));
  Vector v2(4, T(0));

  thrust::copy_n(thrust::make_constant_iterator(thrust::tuple<T, T>(4, 7)),
                 v1.size(),
                 thrust::make_zip_iterator(thrust::make_tuple(v1.begin(), v2.begin())));

  Vector ref1(4, 4);
  Vector ref2(4, 7);

  ASSERT_EQUAL(v1, ref1);
  ASSERT_EQUAL(v2, ref2);
};
DECLARE_VECTOR_UNITTEST(TestCopyNConstantIteratorToZipIterator);

template <typename InputIterator, typename Size, typename OutputIterator>
OutputIterator copy_n(my_system& system, InputIterator, Size, OutputIterator result)
{
  system.validate_dispatch();
  return result;
}

void TestCopyNDispatchExplicit()
{
  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::copy_n(sys, vec.begin(), 1, vec.begin());

  ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestCopyNDispatchExplicit);

template <typename InputIterator, typename Size, typename OutputIterator>
OutputIterator copy_n(my_tag, InputIterator, Size, OutputIterator result)
{
  *result = 13;
  return result;
}

void TestCopyNDispatchImplicit()
{
  thrust::device_vector<int> vec(1);

  thrust::copy_n(thrust::retag<my_tag>(vec.begin()), 1, thrust::retag<my_tag>(vec.begin()));

  ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestCopyNDispatchImplicit);
