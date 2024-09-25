#include <thrust/device_free.h>
#include <thrust/device_malloc.h>
#include <thrust/device_ptr.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/retag.h>

#include <algorithm>

#include <unittest/unittest.h>

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_MSVC(4244 4267) // possible loss of data

template <typename T>
class mark_present_for_each
{
public:
  T* ptr;
  _CCCL_HOST_DEVICE void operator()(T x)
  {
    ptr[(int) x] = 1;
  }
};

template <class Vector>
void TestForEachSimple()
{
  using T = typename Vector::value_type;

  Vector input{3, 2, 3, 4, 6};
  Vector output(7, (T) 0);

  mark_present_for_each<T> f;
  f.ptr = thrust::raw_pointer_cast(output.data());

  typename Vector::iterator result = thrust::for_each(input.begin(), input.end(), f);

  Vector ref{0, 0, 1, 1, 1, 0, 1};
  ASSERT_EQUAL(output, ref);
  ASSERT_EQUAL_QUIET(result, input.end());
}
DECLARE_INTEGRAL_VECTOR_UNITTEST(TestForEachSimple);

template <typename InputIterator, typename Function>
InputIterator for_each(my_system& system, InputIterator first, InputIterator, Function)
{
  system.validate_dispatch();
  return first;
}

void TestForEachDispatchExplicit()
{
  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::for_each(sys, vec.begin(), vec.end(), 0);

  ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestForEachDispatchExplicit);

template <typename InputIterator, typename Function>
InputIterator for_each(my_tag, InputIterator first, InputIterator, Function)
{
  *first = 13;
  return first;
}

void TestForEachDispatchImplicit()
{
  thrust::device_vector<int> vec(1);

  thrust::for_each(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.end()), 0);

  ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestForEachDispatchImplicit);

template <class Vector>
void TestForEachNSimple()
{
  using T = typename Vector::value_type;

  Vector input{3, 2, 3, 4, 6};
  Vector output(7, (T) 0);

  mark_present_for_each<T> f;
  f.ptr = thrust::raw_pointer_cast(output.data());

  typename Vector::iterator result = thrust::for_each_n(input.begin(), input.size(), f);

  Vector ref{0, 0, 1, 1, 1, 0, 1};
  ASSERT_EQUAL(output, ref);
  ASSERT_EQUAL_QUIET(result, input.end());
}
DECLARE_INTEGRAL_VECTOR_UNITTEST(TestForEachNSimple);

template <typename InputIterator, typename Size, typename Function>
InputIterator for_each_n(my_system& system, InputIterator first, Size, Function)
{
  system.validate_dispatch();
  return first;
}

void TestForEachNDispatchExplicit()
{
  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::for_each_n(sys, vec.begin(), vec.size(), 0);

  ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestForEachNDispatchExplicit);

template <typename InputIterator, typename Size, typename Function>
InputIterator for_each_n(my_tag, InputIterator first, Size, Function)
{
  *first = 13;
  return first;
}

void TestForEachNDispatchImplicit()
{
  thrust::device_vector<int> vec(1);

  thrust::for_each_n(thrust::retag<my_tag>(vec.begin()), vec.size(), 0);

  ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestForEachNDispatchImplicit);

void TestForEachSimpleAnySystem()
{
  thrust::device_vector<int> output(7, 0);

  mark_present_for_each<int> f;
  f.ptr = thrust::raw_pointer_cast(output.data());

  thrust::counting_iterator<int> result =
    thrust::for_each(thrust::make_counting_iterator(0), thrust::make_counting_iterator(5), f);

  thrust::device_vector<int> ref{1, 1, 1, 1, 1, 0, 0};
  ASSERT_EQUAL(output, ref);
  ASSERT_EQUAL_QUIET(result, thrust::make_counting_iterator(5));
}
DECLARE_UNITTEST(TestForEachSimpleAnySystem);

void TestForEachNSimpleAnySystem()
{
  thrust::device_vector<int> output(7, 0);

  mark_present_for_each<int> f;
  f.ptr = thrust::raw_pointer_cast(output.data());

  thrust::counting_iterator<int> result = thrust::for_each_n(thrust::make_counting_iterator(0), 5, f);

  thrust::device_vector<int> ref{1, 1, 1, 1, 1, 0, 0};
  ASSERT_EQUAL(output, ref);
  ASSERT_EQUAL_QUIET(result, thrust::make_counting_iterator(5));
}
DECLARE_UNITTEST(TestForEachNSimpleAnySystem);

template <typename T>
void TestForEach(const size_t n)
{
  const size_t output_size = std::min((size_t) 10, 2 * n);

  thrust::host_vector<T> h_input = unittest::random_integers<T>(n);

  for (size_t i = 0; i < n; i++)
  {
    h_input[i] = ((size_t) h_input[i]) % output_size;
  }

  thrust::device_vector<T> d_input = h_input;

  thrust::host_vector<T> h_output(output_size, (T) 0);
  thrust::device_vector<T> d_output(output_size, (T) 0);

  mark_present_for_each<T> h_f;
  mark_present_for_each<T> d_f;
  h_f.ptr = &h_output[0];
  d_f.ptr = (&d_output[0]).get();

  typename thrust::host_vector<T>::iterator h_result = thrust::for_each(h_input.begin(), h_input.end(), h_f);

  typename thrust::device_vector<T>::iterator d_result = thrust::for_each(d_input.begin(), d_input.end(), d_f);

  ASSERT_EQUAL(h_output, d_output);
  ASSERT_EQUAL_QUIET(h_result, h_input.end());
  ASSERT_EQUAL_QUIET(d_result, d_input.end());
}
DECLARE_VARIABLE_UNITTEST(TestForEach);

template <typename T>
void TestForEachN(const size_t n)
{
  const size_t output_size = std::min((size_t) 10, 2 * n);

  thrust::host_vector<T> h_input = unittest::random_integers<T>(n);

  for (size_t i = 0; i < n; i++)
  {
    h_input[i] = ((size_t) h_input[i]) % output_size;
  }

  thrust::device_vector<T> d_input = h_input;

  thrust::host_vector<T> h_output(output_size, (T) 0);
  thrust::device_vector<T> d_output(output_size, (T) 0);

  mark_present_for_each<T> h_f;
  mark_present_for_each<T> d_f;
  h_f.ptr = &h_output[0];
  d_f.ptr = (&d_output[0]).get();

  typename thrust::host_vector<T>::iterator h_result = thrust::for_each_n(h_input.begin(), h_input.size(), h_f);

  typename thrust::device_vector<T>::iterator d_result = thrust::for_each_n(d_input.begin(), d_input.size(), d_f);

  ASSERT_EQUAL(h_output, d_output);
  ASSERT_EQUAL_QUIET(h_result, h_input.end());
  ASSERT_EQUAL_QUIET(d_result, d_input.end());
}
DECLARE_VARIABLE_UNITTEST(TestForEachN);

template <typename T, unsigned int N>
struct SetFixedVectorToConstant
{
  FixedVector<T, N> exemplar;

  SetFixedVectorToConstant(T scalar)
      : exemplar(scalar)
  {}

  _CCCL_HOST_DEVICE void operator()(FixedVector<T, N>& t)
  {
    t = exemplar;
  }
};

template <typename T, unsigned int N>
void _TestForEachWithLargeTypes()
{
  size_t n = (64 * 1024) / sizeof(FixedVector<T, N>);

  thrust::host_vector<FixedVector<T, N>> h_data(n);

  for (size_t i = 0; i < h_data.size(); i++)
  {
    h_data[i] = FixedVector<T, N>(i);
  }

  thrust::device_vector<FixedVector<T, N>> d_data = h_data;

  SetFixedVectorToConstant<T, N> func(123);

  thrust::for_each(h_data.begin(), h_data.end(), func);
  thrust::for_each(d_data.begin(), d_data.end(), func);

  ASSERT_EQUAL_QUIET(h_data, d_data);
}

void TestForEachWithLargeTypes()
{
  _TestForEachWithLargeTypes<int, 1>();
  _TestForEachWithLargeTypes<int, 2>();
  _TestForEachWithLargeTypes<int, 4>();
  _TestForEachWithLargeTypes<int, 8>();
  _TestForEachWithLargeTypes<int, 16>();

  _TestForEachWithLargeTypes<int, 32>(); // fails on Linux 32 w/ gcc 4.1
  _TestForEachWithLargeTypes<int, 64>();
  _TestForEachWithLargeTypes<int, 128>();
  _TestForEachWithLargeTypes<int, 256>();
  _TestForEachWithLargeTypes<int, 512>();

  // XXX parallel_for doens't support large types
  //    _TestForEachWithLargeTypes<int, 1024>();  // fails on Vista 64 w/ VS2008
}
DECLARE_UNITTEST(TestForEachWithLargeTypes);

template <typename T, unsigned int N>
void _TestForEachNWithLargeTypes()
{
  size_t n = (64 * 1024) / sizeof(FixedVector<T, N>);

  thrust::host_vector<FixedVector<T, N>> h_data(n);

  for (size_t i = 0; i < h_data.size(); i++)
  {
    h_data[i] = FixedVector<T, N>(i);
  }

  thrust::device_vector<FixedVector<T, N>> d_data = h_data;

  SetFixedVectorToConstant<T, N> func(123);

  thrust::for_each_n(h_data.begin(), h_data.size(), func);
  thrust::for_each_n(d_data.begin(), d_data.size(), func);

  ASSERT_EQUAL_QUIET(h_data, d_data);
}

void TestForEachNWithLargeTypes()
{
  _TestForEachNWithLargeTypes<int, 1>();
  _TestForEachNWithLargeTypes<int, 2>();
  _TestForEachNWithLargeTypes<int, 4>();
  _TestForEachNWithLargeTypes<int, 8>();
  _TestForEachNWithLargeTypes<int, 16>();

  _TestForEachNWithLargeTypes<int, 32>(); // fails on Linux 32 w/ gcc 4.1
  _TestForEachNWithLargeTypes<int, 64>();
  _TestForEachNWithLargeTypes<int, 128>();
  _TestForEachNWithLargeTypes<int, 256>();
  _TestForEachNWithLargeTypes<int, 512>();

  // XXX parallel_for doens't support large types
  //    _TestForEachNWithLargeTypes<int, 1024>();  // fails on Vista 64 w/ VS2008
}
DECLARE_UNITTEST(TestForEachNWithLargeTypes);

_CCCL_DIAG_POP

struct only_set_when_expected
{
  unsigned long long expected;
  bool* flag;

  _CCCL_DEVICE void operator()(unsigned long long x)
  {
    if (x == expected)
    {
      *flag = true;
    }
  }
};

void TestForEachWithBigIndexesHelper(int magnitude)
{
  thrust::counting_iterator<unsigned long long> begin(0);
  thrust::counting_iterator<unsigned long long> end = begin + (1ull << magnitude);
  ASSERT_EQUAL(thrust::distance(begin, end), 1ll << magnitude);

  thrust::device_ptr<bool> has_executed = thrust::device_malloc<bool>(1);
  *has_executed                         = false;

  only_set_when_expected fn = {(1ull << magnitude) - 1, thrust::raw_pointer_cast(has_executed)};

  thrust::for_each(thrust::device, begin, end, fn);

  bool has_executed_h = *has_executed;
  thrust::device_free(has_executed);

  ASSERT_EQUAL(has_executed_h, true);
}

void TestForEachWithBigIndexes()
{
  TestForEachWithBigIndexesHelper(30);
  TestForEachWithBigIndexesHelper(31);
  TestForEachWithBigIndexesHelper(32);
  TestForEachWithBigIndexesHelper(33);
}
DECLARE_UNITTEST(TestForEachWithBigIndexes);
