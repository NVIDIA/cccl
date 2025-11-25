#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/retag.h>
#include <thrust/tabulate.h>
#include <thrust/transform_scan.h>

#include <algorithm>
#include <numeric>

#include <unittest/unittest.h>

template <typename InputIterator, typename OutputIterator, typename UnaryFunction, typename AssociativeOperator>
OutputIterator transform_inclusive_scan(
  my_system& system, InputIterator, InputIterator, OutputIterator result, UnaryFunction, AssociativeOperator)
{
  system.validate_dispatch();
  return result;
}

void TestTransformInclusiveScanDispatchExplicit()
{
  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::transform_inclusive_scan(sys, vec.begin(), vec.begin(), vec.begin(), 0, 0);

  ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestTransformInclusiveScanDispatchExplicit);

template <typename InputIterator, typename OutputIterator, typename UnaryFunction, typename T, typename AssociativeOperator>
OutputIterator transform_inclusive_scan(
  my_system& system, InputIterator, InputIterator, OutputIterator result, UnaryFunction, T, AssociativeOperator)
{
  system.validate_dispatch();
  return result;
}

void TestTransformInclusiveScanInitDispatchExplicit()
{
  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::transform_inclusive_scan(sys, vec.begin(), vec.begin(), vec.begin(), 0, 0, 0);

  ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestTransformInclusiveScanInitDispatchExplicit);

template <typename InputIterator, typename OutputIterator, typename UnaryFunction, typename AssociativeOperator>
OutputIterator transform_inclusive_scan(
  my_tag, InputIterator, InputIterator, OutputIterator result, UnaryFunction, AssociativeOperator)
{
  *result = 13;
  return result;
}

void TestTransformInclusiveScanDispatchImplicit()
{
  thrust::device_vector<int> vec(1);

  thrust::transform_inclusive_scan(
    thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()), 0, 0);

  ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestTransformInclusiveScanDispatchImplicit);

template <typename InputIterator, typename OutputIterator, typename UnaryFunction, typename T, typename AssociativeOperator>
OutputIterator transform_exclusive_scan(
  my_system& system, InputIterator, InputIterator, OutputIterator result, UnaryFunction, T, AssociativeOperator)
{
  system.validate_dispatch();
  return result;
}

void TestTransformExclusiveScanDispatchExplicit()
{
  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::transform_exclusive_scan(sys, vec.begin(), vec.begin(), vec.begin(), 0, 0, 0);

  ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestTransformExclusiveScanDispatchExplicit);

template <typename InputIterator, typename OutputIterator, typename UnaryFunction, typename T, typename AssociativeOperator>
OutputIterator transform_exclusive_scan(
  my_tag, InputIterator, InputIterator, OutputIterator result, UnaryFunction, T, AssociativeOperator)
{
  *result = 13;
  return result;
}

void TestTransformExclusiveScanDispatchImplicit()
{
  thrust::device_vector<int> vec(1);

  thrust::transform_exclusive_scan(
    thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()), 0, 0, 0);

  ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestTransformExclusiveScanDispatchImplicit);

template <class Vector>
void TestTransformScanSimple()
{
  using T = typename Vector::value_type;

  typename Vector::iterator iter;

  Vector input{1, 3, -2, 4, -5};
  Vector result{-1, -4, -2, -6, -1};
  Vector output(5);

  Vector input_copy(input);

  // inclusive scan
  iter = thrust::transform_inclusive_scan(
    input.begin(), input.end(), output.begin(), ::cuda::std::negate<T>(), ::cuda::std::plus<T>());
  ASSERT_EQUAL(std::size_t(iter - output.begin()), input.size());
  ASSERT_EQUAL(input, input_copy);
  ASSERT_EQUAL(output, result);

  // inclusive scan with 0 init
  iter = thrust::transform_inclusive_scan(
    input.begin(), input.end(), output.begin(), ::cuda::std::negate<T>(), 0, ::cuda::std::plus<T>());
  result = {-1, -4, -2, -6, -1};
  ASSERT_EQUAL(std::size_t(iter - output.begin()), input.size());
  ASSERT_EQUAL(input, input_copy);
  ASSERT_EQUAL(output, result);

  // exclusive scan with 0 init
  iter = thrust::transform_exclusive_scan(
    input.begin(), input.end(), output.begin(), ::cuda::std::negate<T>(), 0, ::cuda::std::plus<T>());
  result = {0, -1, -4, -2, -6};
  ASSERT_EQUAL(std::size_t(iter - output.begin()), input.size());
  ASSERT_EQUAL(input, input_copy);
  ASSERT_EQUAL(output, result);

  // inclusive scan with nonzero init
  iter = thrust::transform_inclusive_scan(
    input.begin(), input.end(), output.begin(), ::cuda::std::negate<T>(), 3, ::cuda::std::plus<T>());
  result = {2, -1, 1, -3, 2};
  ASSERT_EQUAL(std::size_t(iter - output.begin()), input.size());
  ASSERT_EQUAL(input, input_copy);
  ASSERT_EQUAL(output, result);

  // exclusive scan with nonzero init
  iter = thrust::transform_exclusive_scan(
    input.begin(), input.end(), output.begin(), ::cuda::std::negate<T>(), 3, ::cuda::std::plus<T>());
  result = {3, 2, -1, 1, -3};
  ASSERT_EQUAL(std::size_t(iter - output.begin()), input.size());
  ASSERT_EQUAL(input, input_copy);
  ASSERT_EQUAL(output, result);

  // inplace inclusive scan
  input = input_copy;
  iter  = thrust::transform_inclusive_scan(
    input.begin(), input.end(), input.begin(), ::cuda::std::negate<T>(), ::cuda::std::plus<T>());
  result = {-1, -4, -2, -6, -1};
  ASSERT_EQUAL(std::size_t(iter - input.begin()), input.size());
  ASSERT_EQUAL(input, result);

  // inplace inclusive scan with init
  input = input_copy;
  iter  = thrust::transform_inclusive_scan(
    input.begin(), input.end(), input.begin(), ::cuda::std::negate<T>(), 3, ::cuda::std::plus<T>());
  result = {2, -1, 1, -3, 2};
  ASSERT_EQUAL(std::size_t(iter - input.begin()), input.size());
  ASSERT_EQUAL(input, result);

  // inplace exclusive scan with init
  input = input_copy;
  iter  = thrust::transform_exclusive_scan(
    input.begin(), input.end(), input.begin(), ::cuda::std::negate<T>(), 3, ::cuda::std::plus<T>());
  result = {3, 2, -1, 1, -3};
  ASSERT_EQUAL(std::size_t(iter - input.begin()), input.size());
  ASSERT_EQUAL(input, result);
}
DECLARE_INTEGRAL_VECTOR_UNITTEST(TestTransformScanSimple);

struct Record
{
  int number;

  bool operator==(const Record& rhs) const
  {
    return number == rhs.number;
  }
  bool operator!=(const Record& rhs) const
  {
    return !(rhs == *this);
  }
  friend Record operator+(Record lhs, const Record& rhs)
  {
    lhs.number += rhs.number;
    return lhs;
  }
  friend std::ostream& operator<<(std::ostream& os, const Record& record)
  {
    os << "number: " << record.number;
    return os;
  }
};

struct negate
{
  _CCCL_HOST_DEVICE int operator()(Record const& record) const
  {
    return -record.number;
  }
};

void TestTransformInclusiveScanDifferentTypes()
{
  typename thrust::host_vector<int>::iterator h_iter;

  thrust::host_vector<Record> h_input{{1}, {3}, {-2}, {4}, {-5}};
  thrust::host_vector<int> h_output(5);
  thrust::host_vector<int> result{-1, -4, -2, -6, -1};

  thrust::host_vector<Record> input_copy(h_input);

  h_iter = thrust::transform_inclusive_scan(
    h_input.begin(), h_input.end(), h_output.begin(), negate{}, ::cuda::std::plus<int>{});
  ASSERT_EQUAL(std::size_t(h_iter - h_output.begin()), h_input.size());
  ASSERT_EQUAL(h_input, input_copy);
  ASSERT_EQUAL(h_output, result);

  typename thrust::device_vector<int>::iterator d_iter;

  thrust::device_vector<Record> d_input = h_input;
  thrust::device_vector<int> d_output(5);

  d_iter = thrust::transform_inclusive_scan(
    d_input.begin(), d_input.end(), d_output.begin(), negate{}, ::cuda::std::plus<int>{});
  ASSERT_EQUAL(std::size_t(d_iter - d_output.begin()), d_input.size());
  ASSERT_EQUAL(d_input, input_copy);
  ASSERT_EQUAL(d_output, result);
}
DECLARE_UNITTEST(TestTransformInclusiveScanDifferentTypes);

template <typename T>
struct TestTransformScan
{
  void operator()(const size_t n)
  {
    thrust::host_vector<T> h_input   = unittest::random_integers<T>(n);
    thrust::device_vector<T> d_input = h_input;

    thrust::host_vector<T> h_output(n);
    thrust::device_vector<T> d_output(n);

    thrust::transform_inclusive_scan(
      h_input.begin(), h_input.end(), h_output.begin(), ::cuda::std::negate<T>(), ::cuda::std::plus<T>());
    thrust::transform_inclusive_scan(
      d_input.begin(), d_input.end(), d_output.begin(), ::cuda::std::negate<T>(), ::cuda::std::plus<T>());
    ASSERT_EQUAL(d_output, h_output);

    thrust::transform_inclusive_scan(
      h_input.begin(), h_input.end(), h_output.begin(), ::cuda::std::negate<T>(), (T) 11, ::cuda::std::plus<T>());
    thrust::transform_inclusive_scan(
      d_input.begin(), d_input.end(), d_output.begin(), ::cuda::std::negate<T>(), (T) 11, ::cuda::std::plus<T>());
    ASSERT_EQUAL(d_output, h_output);

    thrust::transform_exclusive_scan(
      h_input.begin(), h_input.end(), h_output.begin(), ::cuda::std::negate<T>(), (T) 11, ::cuda::std::plus<T>());
    thrust::transform_exclusive_scan(
      d_input.begin(), d_input.end(), d_output.begin(), ::cuda::std::negate<T>(), (T) 11, ::cuda::std::plus<T>());
    ASSERT_EQUAL(d_output, h_output);

    // in-place scans
    h_output = h_input;
    d_output = d_input;
    thrust::transform_inclusive_scan(
      h_output.begin(), h_output.end(), h_output.begin(), ::cuda::std::negate<T>(), ::cuda::std::plus<T>());
    thrust::transform_inclusive_scan(
      d_output.begin(), d_output.end(), d_output.begin(), ::cuda::std::negate<T>(), ::cuda::std::plus<T>());
    ASSERT_EQUAL(d_output, h_output);

    thrust::transform_inclusive_scan(
      h_output.begin(), h_output.end(), h_output.begin(), ::cuda::std::negate<T>(), (T) 11, ::cuda::std::plus<T>());
    thrust::transform_inclusive_scan(
      d_output.begin(), d_output.end(), d_output.begin(), ::cuda::std::negate<T>(), (T) 11, ::cuda::std::plus<T>());
    ASSERT_EQUAL(d_output, h_output);

    h_output = h_input;
    d_output = d_input;
    thrust::transform_exclusive_scan(
      h_output.begin(), h_output.end(), h_output.begin(), ::cuda::std::negate<T>(), (T) 11, ::cuda::std::plus<T>());
    thrust::transform_exclusive_scan(
      d_output.begin(), d_output.end(), d_output.begin(), ::cuda::std::negate<T>(), (T) 11, ::cuda::std::plus<T>());
    ASSERT_EQUAL(d_output, h_output);
  }
};
VariableUnitTest<TestTransformScan, IntegralTypes> TestTransformScanInstance;

template <class Vector>
void TestTransformScanCountingIterator()
{
  using T     = typename Vector::value_type;
  using space = typename thrust::iterator_system<typename Vector::iterator>::type;

  thrust::counting_iterator<T, space> first(1);

  Vector result(3);

  thrust::transform_inclusive_scan(first, first + 3, result.begin(), ::cuda::std::negate<T>(), ::cuda::std::plus<T>());

  Vector ref{-1, -3, -6};
  ASSERT_EQUAL(result, ref);
}
DECLARE_INTEGRAL_VECTOR_UNITTEST(TestTransformScanCountingIterator);

template <typename T>
struct TestTransformScanToDiscardIterator
{
  void operator()(const size_t n)
  {
    thrust::host_vector<T> h_input   = unittest::random_integers<T>(n);
    thrust::device_vector<T> d_input = h_input;

    thrust::discard_iterator<> reference(n);

    thrust::discard_iterator<> h_result = thrust::transform_inclusive_scan(
      h_input.begin(), h_input.end(), thrust::make_discard_iterator(), ::cuda::std::negate<T>(), ::cuda::std::plus<T>());

    thrust::discard_iterator<> d_result = thrust::transform_inclusive_scan(
      d_input.begin(), d_input.end(), thrust::make_discard_iterator(), ::cuda::std::negate<T>(), ::cuda::std::plus<T>());
    ASSERT_EQUAL_QUIET(reference, h_result);
    ASSERT_EQUAL_QUIET(reference, d_result);

    h_result = thrust::transform_inclusive_scan(
      h_input.begin(),
      h_input.end(),
      thrust::make_discard_iterator(),
      ::cuda::std::negate<T>(),
      (T) 11,
      ::cuda::std::plus<T>());

    d_result = thrust::transform_inclusive_scan(
      d_input.begin(),
      d_input.end(),
      thrust::make_discard_iterator(),
      ::cuda::std::negate<T>(),
      (T) 11,
      ::cuda::std::plus<T>());

    h_result = thrust::transform_exclusive_scan(
      h_input.begin(),
      h_input.end(),
      thrust::make_discard_iterator(),
      ::cuda::std::negate<T>(),
      (T) 11,
      ::cuda::std::plus<T>());

    d_result = thrust::transform_exclusive_scan(
      d_input.begin(),
      d_input.end(),
      thrust::make_discard_iterator(),
      ::cuda::std::negate<T>(),
      (T) 11,
      ::cuda::std::plus<T>());

    ASSERT_EQUAL_QUIET(reference, h_result);
    ASSERT_EQUAL_QUIET(reference, d_result);
  }
};
VariableUnitTest<TestTransformScanToDiscardIterator, IntegralTypes> TestTransformScanToDiscardIteratorInstance;

// Regression test for https://github.com/NVIDIA/thrust/issues/1332
// The issue was the internal transform_input_iterator_t created by the
// transform_inclusive_scan implementation was instantiated using a reference
// type for the value_type.
template <typename T>
void TestValueCategoryDeduction()
{
  thrust::device_vector<T> vec;

  T a_h[10] = {5, 0, 5, 8, 6, 7, 5, 3, 0, 9};
  vec.assign((T*) a_h, a_h + 10);

  thrust::transform_inclusive_scan(
    thrust::device, vec.cbegin(), vec.cend(), vec.begin(), ::cuda::std::identity{}, ::cuda::maximum<>{});

  ASSERT_EQUAL((thrust::device_vector<T>{5, 5, 5, 8, 8, 8, 8, 8, 8, 9}), vec);

  vec.assign((T*) a_h, a_h + 10);

  thrust::transform_inclusive_scan(
    thrust::device, vec.cbegin(), vec.cend(), vec.begin(), ::cuda::std::identity{}, T{}, ::cuda::maximum<>{});

  ASSERT_EQUAL((thrust::device_vector<T>{5, 5, 5, 8, 8, 8, 8, 8, 8, 9}), vec);

  vec.assign((T*) a_h, a_h + 10);
  thrust::transform_exclusive_scan(
    thrust::device, vec.cbegin(), vec.cend(), vec.begin(), ::cuda::std::identity{}, T{}, ::cuda::maximum<>{});

  ASSERT_EQUAL((thrust::device_vector<T>{0, 5, 5, 5, 8, 8, 8, 8, 8, 8}), vec);
}
DECLARE_GENERIC_UNITTEST(TestValueCategoryDeduction);

// User-defined Int type with poison default constructor
struct TransformInt
{
  int value;

  _CCCL_HOST_DEVICE TransformInt()
      : value(999)
  {}
  _CCCL_HOST_DEVICE TransformInt(int v)
      : value(v)
  {}

  _CCCL_HOST_DEVICE friend TransformInt operator+(TransformInt a, TransformInt b)
  {
    return TransformInt(a.value + b.value);
  }
  _CCCL_HOST_DEVICE friend TransformInt operator*(TransformInt a, TransformInt b)
  {
    return TransformInt(a.value * b.value);
  }
  _CCCL_HOST_DEVICE friend bool operator==(TransformInt a, TransformInt b)
  {
    return a.value == b.value;
  }
  friend std::ostream& operator<<(std::ostream& os, TransformInt i)
  {
    return os << i.value;
  }
};

// Test edge cases for parallel transform_scan with non-additive operations
void TestTransformScanEdgeCases()
{
  // Test 1: Large array with transform_inclusive_scan, multiplies, and init
  {
    const int n = 10000;
    thrust::device_vector<int> d_input(n);
    using thrust::placeholders::_1;
    thrust::tabulate(d_input.begin(), d_input.end(), (_1 % 5) + 1);

    thrust::device_vector<int> d_output(n);
    auto r = thrust::transform_inclusive_scan(
      d_input.begin(), d_input.end(), d_output.begin(), ::cuda::std::negate<int>(), 2, ::cuda::std::multiplies<>{});
    ASSERT_EQUAL((d_output.end() == r), true);

    // Verify with host scan
    thrust::host_vector<int> h_input = d_input;
    thrust::host_vector<int> h_output(n);
    thrust::transform_inclusive_scan(
      h_input.begin(), h_input.end(), h_output.begin(), ::cuda::std::negate<int>(), 2, ::cuda::std::multiplies<>{});

    ASSERT_EQUAL(d_output, h_output);
  }

  // Test 2: transform_exclusive_scan with multiplies on large array
  {
    const int n = 10000;
    thrust::device_vector<int> d_input(n);
    using thrust::placeholders::_1;
    thrust::tabulate(d_input.begin(), d_input.end(), (_1 % 3) + 1);

    thrust::device_vector<int> d_output(n);
    auto r = thrust::transform_exclusive_scan(
      d_input.begin(), d_input.end(), d_output.begin(), ::cuda::std::negate<int>(), 5, ::cuda::std::multiplies<>{});
    ASSERT_EQUAL((d_output.end() == r), true);

    thrust::host_vector<int> h_input = d_input;
    thrust::host_vector<int> h_output(n);
    thrust::transform_exclusive_scan(
      h_input.begin(), h_input.end(), h_output.begin(), ::cuda::std::negate<int>(), 5, ::cuda::std::multiplies<>{});

    ASSERT_EQUAL(d_output, h_output);
  }

  // Test 3: User-defined type with poison default constructor
  {
    thrust::device_vector<TransformInt> vec(5, TransformInt{1});
    thrust::transform_exclusive_scan(
      vec.begin(),
      vec.end(),
      vec.begin(),
      ::cuda::std::identity{},
      TransformInt{100},
      ::cuda::std::plus<TransformInt>());

    ASSERT_EQUAL(vec,
                 (thrust::device_vector<TransformInt>{
                   TransformInt{100}, TransformInt{101}, TransformInt{102}, TransformInt{103}, TransformInt{104}}));
  }

  // Test 4: In-place transform_exclusive_scan with multiplies
  {
    thrust::device_vector<int> vec = {2, 3, 4, 5};
    auto r                         = thrust::transform_exclusive_scan(
      vec.begin(), vec.end(), vec.begin(), ::cuda::std::negate<int>(), 10, ::cuda::std::multiplies<>{});
    ASSERT_EQUAL((vec.end() == r), true);

    thrust::device_vector<int> expected = {10, -20, 60, -240};
    ASSERT_EQUAL(vec, expected);
  }

  // Test 5: Boundary case at threshold (1024 elements)
  {
    const int n = 1024;
    thrust::device_vector<int> d_input(n);
    using thrust::placeholders::_1;
    thrust::tabulate(d_input.begin(), d_input.end(), _1 + 1);

    thrust::device_vector<int> d_output(n);
    auto r = thrust::transform_inclusive_scan(
      d_input.begin(), d_input.end(), d_output.begin(), thrust::square<int>(), 1, ::cuda::std::multiplies<>{});
    ASSERT_EQUAL((d_output.end() == r), true);

    thrust::host_vector<int> h_input = d_input;
    thrust::host_vector<int> h_output(n);
    thrust::transform_inclusive_scan(
      h_input.begin(), h_input.end(), h_output.begin(), thrust::square<int>(), 1, ::cuda::std::multiplies<>{});

    ASSERT_EQUAL(d_output, h_output);
  }

  // Test 6: Very small array (2 elements)
  {
    thrust::device_vector<int> d_input = {3, 7};
    thrust::device_vector<int> d_output(2);

    auto r = thrust::transform_inclusive_scan(
      d_input.begin(), d_input.end(), d_output.begin(), ::cuda::std::negate<int>(), 2, ::cuda::std::multiplies<>{});
    ASSERT_EQUAL((d_output.end() == r), true);

    thrust::device_vector<int> expected = {-6, 42};
    ASSERT_EQUAL(d_output, expected);
  }
}
DECLARE_UNITTEST(TestTransformScanEdgeCases);
