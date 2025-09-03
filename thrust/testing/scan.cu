#include <thrust/detail/config.h>

#include <thrust/device_free.h>
#include <thrust/device_malloc.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/retag.h>
#include <thrust/scan.h>

#include <cuda/functional>
#include <cuda/std/array>

#include <unittest/unittest.h>

template <class Vector>
void TestScanSimple()
{
  using T = typename Vector::value_type;
  typename Vector::iterator iter;

  Vector input(5);
  Vector result(5);
  Vector output(5);

  input = {1, 3, -2, 4, -5};
  Vector input_copy(input);

  // inclusive scan
  iter   = thrust::inclusive_scan(input.begin(), input.end(), output.begin());
  result = {1, 4, 2, 6, 1};
  ASSERT_EQUAL(std::size_t(iter - output.begin()), input.size());
  ASSERT_EQUAL(input, input_copy);
  ASSERT_EQUAL(output, result);

  // exclusive scan
  iter   = thrust::exclusive_scan(input.begin(), input.end(), output.begin(), T(0));
  result = {0, 1, 4, 2, 6};
  ASSERT_EQUAL(std::size_t(iter - output.begin()), input.size());
  ASSERT_EQUAL(input, input_copy);
  ASSERT_EQUAL(output, result);

  // exclusive scan with init
  iter   = thrust::exclusive_scan(input.begin(), input.end(), output.begin(), T(3));
  result = {3, 4, 7, 5, 9};
  ASSERT_EQUAL(std::size_t(iter - output.begin()), input.size());
  ASSERT_EQUAL(input, input_copy);
  ASSERT_EQUAL(output, result);

  // inclusive scan with op
  iter   = thrust::inclusive_scan(input.begin(), input.end(), output.begin(), ::cuda::std::plus<T>());
  result = {1, 4, 2, 6, 1};
  ASSERT_EQUAL(std::size_t(iter - output.begin()), input.size());
  ASSERT_EQUAL(input, input_copy);
  ASSERT_EQUAL(output, result);

  // inclusive scan with init and op
  iter   = thrust::inclusive_scan(input.begin(), input.end(), output.begin(), T(-1), ::cuda::std::multiplies<T>());
  result = {-1, -3, 6, 24, -120};
  ASSERT_EQUAL(std::size_t(iter - output.begin()), input.size());
  ASSERT_EQUAL(input, input_copy);
  ASSERT_EQUAL(output, result);

  // exclusive scan with init and op
  iter   = thrust::exclusive_scan(input.begin(), input.end(), output.begin(), T(3), ::cuda::std::plus<T>());
  result = {3, 4, 7, 5, 9};
  ASSERT_EQUAL(std::size_t(iter - output.begin()), input.size());
  ASSERT_EQUAL(input, input_copy);
  ASSERT_EQUAL(output, result);

  // inplace inclusive scan
  input  = input_copy;
  iter   = thrust::inclusive_scan(input.begin(), input.end(), input.begin());
  result = {1, 4, 2, 6, 1};
  ASSERT_EQUAL(std::size_t(iter - input.begin()), input.size());
  ASSERT_EQUAL(input, result);

  // inplace inclusive scan with init and op
  input  = input_copy;
  iter   = thrust::inclusive_scan(input.begin(), input.end(), input.begin(), T(3), ::cuda::std::plus<T>());
  result = {4, 7, 5, 9, 4};
  ASSERT_EQUAL(std::size_t(iter - input.begin()), input.size());
  ASSERT_EQUAL(input, result);

  // inplace exclusive scan with init
  input  = input_copy;
  iter   = thrust::exclusive_scan(input.begin(), input.end(), input.begin(), T(3));
  result = {3, 4, 7, 5, 9};
  ASSERT_EQUAL(std::size_t(iter - input.begin()), input.size());
  ASSERT_EQUAL(input, result);

  // inplace exclusive scan with implicit init=0
  input  = input_copy;
  iter   = thrust::exclusive_scan(input.begin(), input.end(), input.begin());
  result = {0, 1, 4, 2, 6};
  ASSERT_EQUAL(std::size_t(iter - input.begin()), input.size());
  ASSERT_EQUAL(input, result);
}
DECLARE_VECTOR_UNITTEST(TestScanSimple);

template <typename InputIterator, typename OutputIterator>
OutputIterator inclusive_scan(my_system& system, InputIterator, InputIterator, OutputIterator result)
{
  system.validate_dispatch();
  return result;
}

void TestInclusiveScanDispatchExplicit()
{
  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::inclusive_scan(sys, vec.begin(), vec.begin(), vec.begin());

  ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestInclusiveScanDispatchExplicit);

template <typename InputIterator, typename OutputIterator>
OutputIterator inclusive_scan(my_tag, InputIterator, InputIterator, OutputIterator result)
{
  *result = 13;
  return result;
}

void TestInclusiveScanDispatchImplicit()
{
  thrust::device_vector<int> vec(1);

  thrust::inclusive_scan(
    thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()));

  ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestInclusiveScanDispatchImplicit);

template <typename InputIterator, typename OutputIterator>
OutputIterator exclusive_scan(my_system& system, InputIterator, InputIterator, OutputIterator result)
{
  system.validate_dispatch();
  return result;
}

void TestExclusiveScanDispatchExplicit()
{
  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::exclusive_scan(sys, vec.begin(), vec.begin(), vec.begin());

  ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestExclusiveScanDispatchExplicit);

template <typename InputIterator, typename OutputIterator>
OutputIterator exclusive_scan(my_tag, InputIterator, InputIterator, OutputIterator result)
{
  *result = 13;
  return result;
}

void TestExclusiveScanDispatchImplicit()
{
  thrust::device_vector<int> vec(1);

  thrust::exclusive_scan(
    thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()));

  ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestExclusiveScanDispatchImplicit);

void TestInclusiveScan32()
{
  using T  = int;
  size_t n = 32;

  thrust::host_vector<T> h_input   = unittest::random_integers<T>(n);
  thrust::device_vector<T> d_input = h_input;

  thrust::host_vector<T> h_output(n);
  thrust::device_vector<T> d_output(n);

  thrust::inclusive_scan(h_input.begin(), h_input.end(), h_output.begin());
  thrust::inclusive_scan(d_input.begin(), d_input.end(), d_output.begin());

  ASSERT_EQUAL(d_output, h_output);
}
DECLARE_UNITTEST(TestInclusiveScan32);

void TestExclusiveScan32()
{
  using T  = int;
  size_t n = 32;
  T init   = 13;

  thrust::host_vector<T> h_input   = unittest::random_integers<T>(n);
  thrust::device_vector<T> d_input = h_input;

  thrust::host_vector<T> h_output(n);
  thrust::device_vector<T> d_output(n);

  thrust::exclusive_scan(h_input.begin(), h_input.end(), h_output.begin(), init);
  thrust::exclusive_scan(d_input.begin(), d_input.end(), d_output.begin(), init);

  ASSERT_EQUAL(d_output, h_output);
}
DECLARE_UNITTEST(TestExclusiveScan32);

template <class IntVector, class FloatVector>
void TestScanMixedTypes()
{
  // make sure we get types for default args and operators correct
  IntVector int_input{1, 2, 3, 4};
  FloatVector float_input{1.5, 2.5, 3.5, 4.5};
  IntVector int_output(4);
  FloatVector float_output(4);

  // float -> int should use plus<void> operator and float accumulator by default
  thrust::inclusive_scan(float_input.begin(), float_input.end(), int_output.begin());
  ASSERT_EQUAL(int_output[0], 1); // in: 1.5 accum: 1.5f out: 1
  ASSERT_EQUAL(int_output[1], 4); // in: 2.5 accum: 4.0f out: 4
  ASSERT_EQUAL(int_output[2], 7); // in: 3.5 accum: 7.5f out: 7
  ASSERT_EQUAL(int_output[3], 12); // in: 4.5 accum: 12.f out: 12

  // float -> float with plus<int> operator (float accumulator)
  thrust::inclusive_scan(float_input.begin(), float_input.end(), float_output.begin(), ::cuda::std::plus<int>());
  ASSERT_EQUAL(float_output[0], 1.5f); // in: 1.5 accum: 1.5f out: 1.5f
  ASSERT_EQUAL(float_output[1], 3.0f); // in: 2.5 accum: 3.0f out: 3.0f
  ASSERT_EQUAL(float_output[2], 6.0f); // in: 3.5 accum: 6.0f out: 6.0f
  ASSERT_EQUAL(float_output[3], 10.0f); // in: 4.5 accum: 10.f out: 10.f

  // float -> int should use plus<void> operator and float accumulator by default
  thrust::exclusive_scan(float_input.begin(), float_input.end(), int_output.begin());
  ASSERT_EQUAL(int_output[0], 0); // out: 0.0f  in: 1.5 accum: 1.5f
  ASSERT_EQUAL(int_output[1], 1); // out: 1.5f  in: 2.5 accum: 4.0f
  ASSERT_EQUAL(int_output[2], 4); // out: 4.0f  in: 3.5 accum: 7.5f
  ASSERT_EQUAL(int_output[3], 7); // out: 7.5f  in: 4.5 accum: 12.f

  // float -> int should use plus<> operator and float accumulator by default
  thrust::exclusive_scan(float_input.begin(), float_input.end(), int_output.begin(), (float) 5.5);
  ASSERT_EQUAL(int_output[0], 5); // out: 5.5f  in: 1.5 accum: 7.0f
  ASSERT_EQUAL(int_output[1], 7); // out: 7.0f  in: 2.5 accum: 9.5f
  ASSERT_EQUAL(int_output[2], 9); // out: 9.5f  in: 3.5 accum: 13.0f
  ASSERT_EQUAL(int_output[3], 13); // out: 13.f  in: 4.5 accum: 17.4f

  // int -> float should use using plus<> operator and int accumulator by default
  thrust::inclusive_scan(int_input.begin(), int_input.end(), float_output.begin());
  ASSERT_EQUAL(float_output[0], 1.f); // in: 1 accum: 1  out: 1
  ASSERT_EQUAL(float_output[1], 3.f); // in: 2 accum: 3  out: 3
  ASSERT_EQUAL(float_output[2], 6.f); // in: 3 accum: 6  out: 6
  ASSERT_EQUAL(float_output[3], 10.f); // in: 4 accum: 10 out: 10

  // int -> float + float init_value should use using plus<> operator and
  // float accumulator by default
  thrust::exclusive_scan(int_input.begin(), int_input.end(), float_output.begin(), (float) 5.5);
  ASSERT_EQUAL(float_output[0], 5.5f); // out: 5.5f  in: 1 accum: 6.5f
  ASSERT_EQUAL(float_output[1], 6.5f); // out: 6.0f  in: 2 accum: 8.5f
  ASSERT_EQUAL(float_output[2], 8.5f); // out: 8.0f  in: 3 accum: 11.5f
  ASSERT_EQUAL(float_output[3], 11.5f); // out: 11.f  in: 4 accum: 15.5f
}
void TestScanMixedTypesHost()
{
  TestScanMixedTypes<thrust::host_vector<int>, thrust::host_vector<float>>();
}
DECLARE_UNITTEST(TestScanMixedTypesHost);
void TestScanMixedTypesDevice()
{
  TestScanMixedTypes<thrust::device_vector<int>, thrust::device_vector<float>>();
}
DECLARE_UNITTEST(TestScanMixedTypesDevice);

template <typename T>
struct TestScanWithOperator
{
  void operator()(const size_t n)
  {
    thrust::host_vector<T> h_input   = unittest::random_integers<T>(n);
    thrust::device_vector<T> d_input = h_input;

    thrust::host_vector<T> h_output(n);
    thrust::device_vector<T> d_output(n);

    thrust::inclusive_scan(h_input.begin(), h_input.end(), h_output.begin(), cuda::maximum<T>{});
    thrust::inclusive_scan(d_input.begin(), d_input.end(), d_output.begin(), cuda::maximum<T>{});
    ASSERT_EQUAL(d_output, h_output);

    thrust::exclusive_scan(h_input.begin(), h_input.end(), h_output.begin(), T(13), cuda::maximum<T>{});
    thrust::exclusive_scan(d_input.begin(), d_input.end(), d_output.begin(), T(13), cuda::maximum<T>{});
    ASSERT_EQUAL(d_output, h_output);
  }
};
VariableUnitTest<TestScanWithOperator, SignedIntegralTypes> TestScanWithOperatorInstance;

template <typename T>
struct TestScanWithOperatorToDiscardIterator
{
  void operator()(const size_t n)
  {
    thrust::host_vector<T> h_input   = unittest::random_integers<T>(n);
    thrust::device_vector<T> d_input = h_input;

    thrust::discard_iterator<> reference(n);

    thrust::discard_iterator<> h_result =
      thrust::inclusive_scan(h_input.begin(), h_input.end(), thrust::make_discard_iterator(), cuda::maximum<T>{});

    thrust::discard_iterator<> d_result =
      thrust::inclusive_scan(d_input.begin(), d_input.end(), thrust::make_discard_iterator(), cuda::maximum<T>{});

    ASSERT_EQUAL_QUIET(reference, h_result);
    ASSERT_EQUAL_QUIET(reference, d_result);

    h_result = thrust::exclusive_scan(
      h_input.begin(), h_input.end(), thrust::make_discard_iterator(), T(13), cuda::maximum<T>{});

    d_result = thrust::exclusive_scan(
      d_input.begin(), d_input.end(), thrust::make_discard_iterator(), T(13), cuda::maximum<T>{});

    ASSERT_EQUAL_QUIET(reference, h_result);
    ASSERT_EQUAL_QUIET(reference, d_result);
  }
};
VariableUnitTest<TestScanWithOperatorToDiscardIterator,
                 unittest::type_list<unittest::int8_t, unittest::int16_t, unittest::int32_t>>
  TestScanWithOperatorToDiscardIteratorInstance;

template <typename T>
struct TestScan
{
  void operator()(const size_t n)
  {
    thrust::host_vector<T> h_input   = unittest::random_integers<T>(n);
    thrust::device_vector<T> d_input = h_input;

    thrust::host_vector<T> h_output(n);
    thrust::device_vector<T> d_output(n);

    thrust::inclusive_scan(h_input.begin(), h_input.end(), h_output.begin());
    thrust::inclusive_scan(d_input.begin(), d_input.end(), d_output.begin());
    ASSERT_EQUAL(d_output, h_output);

    thrust::exclusive_scan(h_input.begin(), h_input.end(), h_output.begin());
    thrust::exclusive_scan(d_input.begin(), d_input.end(), d_output.begin());
    ASSERT_EQUAL(d_output, h_output);

    thrust::exclusive_scan(h_input.begin(), h_input.end(), h_output.begin(), (T) 11);
    thrust::exclusive_scan(d_input.begin(), d_input.end(), d_output.begin(), (T) 11);
    ASSERT_EQUAL(d_output, h_output);

    // in-place scans
    h_output = h_input;
    d_output = d_input;
    thrust::inclusive_scan(h_output.begin(), h_output.end(), h_output.begin());
    thrust::inclusive_scan(d_output.begin(), d_output.end(), d_output.begin());
    ASSERT_EQUAL(d_output, h_output);

    h_output = h_input;
    d_output = d_input;
    thrust::exclusive_scan(h_output.begin(), h_output.end(), h_output.begin());
    thrust::exclusive_scan(d_output.begin(), d_output.end(), d_output.begin());
    ASSERT_EQUAL(d_output, h_output);
  }
};
VariableUnitTest<TestScan, IntegralTypes> TestScanInstance;

template <typename T>
struct TestScanToDiscardIterator
{
  void operator()(const size_t n)
  {
    thrust::host_vector<T> h_input   = unittest::random_integers<T>(n);
    thrust::device_vector<T> d_input = h_input;

    thrust::discard_iterator<> h_result =
      thrust::inclusive_scan(h_input.begin(), h_input.end(), thrust::make_discard_iterator());

    thrust::discard_iterator<> d_result =
      thrust::inclusive_scan(d_input.begin(), d_input.end(), thrust::make_discard_iterator());

    thrust::discard_iterator<> reference(n);

    ASSERT_EQUAL_QUIET(reference, h_result);
    ASSERT_EQUAL_QUIET(reference, d_result);

    h_result = thrust::exclusive_scan(h_input.begin(), h_input.end(), thrust::make_discard_iterator(), (T) 11);

    d_result = thrust::exclusive_scan(d_input.begin(), d_input.end(), thrust::make_discard_iterator(), (T) 11);

    ASSERT_EQUAL_QUIET(reference, h_result);
    ASSERT_EQUAL_QUIET(reference, d_result);
  }
};
VariableUnitTest<TestScanToDiscardIterator, unittest::type_list<unittest::int8_t, unittest::int16_t, unittest::int32_t>>
  TestScanToDiscardIteratorInstance;

void TestScanMixedTypes()
{
  const unsigned int n = 113;

  thrust::host_vector<unsigned int> h_input = unittest::random_integers<unsigned int>(n);
  for (size_t i = 0; i < n; i++)
  {
    h_input[i] %= 10;
  }
  thrust::device_vector<unsigned int> d_input = h_input;

  thrust::host_vector<float> h_float_output(n);
  thrust::device_vector<float> d_float_output(n);
  thrust::host_vector<int> h_int_output(n);
  thrust::device_vector<int> d_int_output(n);

  // mixed input/output types
  thrust::inclusive_scan(h_input.begin(), h_input.end(), h_float_output.begin());
  thrust::inclusive_scan(d_input.begin(), d_input.end(), d_float_output.begin());
  ASSERT_EQUAL(d_float_output, h_float_output);

  thrust::exclusive_scan(h_input.begin(), h_input.end(), h_float_output.begin(), (float) 3.5);
  thrust::exclusive_scan(d_input.begin(), d_input.end(), d_float_output.begin(), (float) 3.5);
  ASSERT_EQUAL(d_float_output, h_float_output);

  thrust::exclusive_scan(h_input.begin(), h_input.end(), h_float_output.begin(), (int) 3);
  thrust::exclusive_scan(d_input.begin(), d_input.end(), d_float_output.begin(), (int) 3);
  ASSERT_EQUAL(d_float_output, h_float_output);

  thrust::exclusive_scan(h_input.begin(), h_input.end(), h_int_output.begin(), (int) 3);
  thrust::exclusive_scan(d_input.begin(), d_input.end(), d_int_output.begin(), (int) 3);
  ASSERT_EQUAL(d_int_output, h_int_output);

  thrust::exclusive_scan(h_input.begin(), h_input.end(), h_int_output.begin(), (float) 3.5);
  thrust::exclusive_scan(d_input.begin(), d_input.end(), d_int_output.begin(), (float) 3.5);
  ASSERT_EQUAL(d_int_output, h_int_output);
}
DECLARE_UNITTEST(TestScanMixedTypes);

template <typename T, unsigned int N>
void _TestScanWithLargeTypes()
{
  size_t n = (1024 * 1024) / sizeof(FixedVector<T, N>);

  thrust::host_vector<FixedVector<T, N>> h_input(n);
  thrust::host_vector<FixedVector<T, N>> h_output(n);

  for (size_t i = 0; i < h_input.size(); i++)
  {
    h_input[i] = FixedVector<T, N>(static_cast<T>(i));
  }

  thrust::device_vector<FixedVector<T, N>> d_input = h_input;
  thrust::device_vector<FixedVector<T, N>> d_output(n);

  thrust::inclusive_scan(h_input.begin(), h_input.end(), h_output.begin());
  thrust::inclusive_scan(d_input.begin(), d_input.end(), d_output.begin());

  ASSERT_EQUAL_QUIET(h_output, d_output);

  thrust::exclusive_scan(h_input.begin(), h_input.end(), h_output.begin(), FixedVector<T, N>(0));
  thrust::exclusive_scan(d_input.begin(), d_input.end(), d_output.begin(), FixedVector<T, N>(0));

  ASSERT_EQUAL_QUIET(h_output, d_output);
}

void TestScanWithLargeTypes()
{
  _TestScanWithLargeTypes<int, 1>();

#if !defined(__QNX__)
  _TestScanWithLargeTypes<int, 8>();
  _TestScanWithLargeTypes<int, 64>();
#else
  KNOWN_FAILURE;
#endif
}
DECLARE_UNITTEST(TestScanWithLargeTypes);

template <typename T>
struct plus_mod3
{
  T* table;

  plus_mod3(T* table)
      : table(table)
  {}

  _CCCL_HOST_DEVICE T operator()(T a, T b)
  {
    return table[(int) (a + b)];
  }
};

template <typename Vector>
void TestInclusiveScanWithIndirection()
{
  // add numbers modulo 3 with external lookup table
  using T = typename Vector::value_type;

  Vector data{0, 1, 2, 1, 2, 0, 1};
  Vector table{0, 1, 2, 0, 1, 2};
  thrust::inclusive_scan(data.begin(), data.end(), data.begin(), plus_mod3<T>(thrust::raw_pointer_cast(&table[0])));

  ASSERT_EQUAL(data, (Vector{0, 1, 0, 1, 0, 0, 1}));
}
DECLARE_INTEGRAL_VECTOR_UNITTEST(TestInclusiveScanWithIndirection);

template <typename T>
struct const_ref_plus_mod3
{
  T* table;

  const_ref_plus_mod3(T* table)
      : table(table)
  {}

  _CCCL_HOST_DEVICE const T& operator()(T a, T b)
  {
    return table[(int) (a + b)];
  }
};

template <typename Vector>
void TestInclusiveScanWithConstAccumulator()
{
  // add numbers modulo 3 with external lookup table
  using T = typename Vector::value_type;

  Vector data{0, 1, 2, 1, 2, 0, 1};
  Vector table{0, 1, 2, 0, 1, 2};
  thrust::inclusive_scan(
    data.begin(), data.end(), data.begin(), const_ref_plus_mod3<T>(thrust::raw_pointer_cast(&table[0])));

  ASSERT_EQUAL(data, (Vector{0, 1, 0, 1, 0, 0, 1}));
}
DECLARE_INTEGRAL_VECTOR_UNITTEST(TestInclusiveScanWithConstAccumulator);

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

namespace std
{
template <>
struct iterator_traits<only_set_when_expected_it>
{
  using value_type      = long long;
  using reference       = only_set_when_expected_it;
  using difference_type = ::cuda::std::ptrdiff_t;
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

void TestInclusiveScanWithBigIndexesHelper(int magnitude)
{
  thrust::constant_iterator<long long> begin(1);
  thrust::constant_iterator<long long> end = begin + (1ll << magnitude);
  ASSERT_EQUAL(::cuda::std::distance(begin, end), 1ll << magnitude);

  thrust::device_ptr<bool> has_executed = thrust::device_malloc<bool>(1);
  *has_executed                         = false;

  only_set_when_expected_it out = {(1ll << magnitude), thrust::raw_pointer_cast(has_executed)};

  thrust::inclusive_scan(thrust::device, begin, end, out);

  bool has_executed_h = *has_executed;
  thrust::device_free(has_executed);

  ASSERT_EQUAL(has_executed_h, true);
}

void TestInclusiveScanWithBigIndexes()
{
  TestInclusiveScanWithBigIndexesHelper(30);
  TestInclusiveScanWithBigIndexesHelper(31);
#ifndef THRUST_FORCE_32_BIT_OFFSET_TYPE
  TestInclusiveScanWithBigIndexesHelper(32);
  TestInclusiveScanWithBigIndexesHelper(33);
#endif
}

DECLARE_UNITTEST(TestInclusiveScanWithBigIndexes);

void TestExclusiveScanWithBigIndexesHelper(int magnitude)
{
  thrust::constant_iterator<long long> begin(1);
  thrust::constant_iterator<long long> end = begin + (1ll << magnitude);
  ASSERT_EQUAL(::cuda::std::distance(begin, end), 1ll << magnitude);

  thrust::device_ptr<bool> has_executed = thrust::device_malloc<bool>(1);
  *has_executed                         = false;

  only_set_when_expected_it out = {(1ll << magnitude) - 1, thrust::raw_pointer_cast(has_executed)};

  thrust::exclusive_scan(thrust::device, begin, end, out, 0ll);

  bool has_executed_h = *has_executed;
  thrust::device_free(has_executed);

  ASSERT_EQUAL(has_executed_h, true);
}

void TestExclusiveScanWithBigIndexes()
{
  TestExclusiveScanWithBigIndexesHelper(30);
  TestExclusiveScanWithBigIndexesHelper(31);
#ifndef THRUST_FORCE_32_BIT_OFFSET_TYPE
  TestExclusiveScanWithBigIndexesHelper(32);
  TestExclusiveScanWithBigIndexesHelper(33);
#endif
}

DECLARE_UNITTEST(TestExclusiveScanWithBigIndexes);

struct Int
{
  int i{};
  _CCCL_HOST_DEVICE explicit Int(int num)
      : i(num)
  {}
  _CCCL_HOST_DEVICE Int()
      : i{}
  {}
  _CCCL_HOST_DEVICE Int operator+(Int const& o) const
  {
    return Int{this->i + o.i};
  }
};

void TestInclusiveScanWithUserDefinedType()
{
  thrust::device_vector<Int> vec(5, Int{1});

  thrust::inclusive_scan(thrust::device, vec.cbegin(), vec.cend(), vec.begin());

  ASSERT_EQUAL(static_cast<Int>(vec.back()).i, 5);
}
DECLARE_UNITTEST(TestInclusiveScanWithUserDefinedType);

// Represents a permutation as a tuple of integers, see also: https://en.wikipedia.org/wiki/Permutation
// We need a distinct type (instead of an alias) for operator<< to be found via ADL
struct permutation_t : ::cuda::std::array<int, 5>
{
  permutation_t() = default;

  constexpr _CCCL_HOST_DEVICE permutation_t(int a, int b, int c, int d, int e)
      : ::cuda::std::array<int, 5>{a, b, c, d, e}
  {}

  friend std::ostream& operator<<(std::ostream& os, const permutation_t& p)
  {
    os << '{';
    for (std::size_t i = 0; i < p.size(); i++)
    {
      if (i > 0)
      {
        os << ", ";
      }
      os << p[i];
    }
    return os << '}';
  }
};

// Composes two permutations. This operation is associative, but not commutative.
struct composition_op_t
{
  _CCCL_HOST_DEVICE permutation_t operator()(permutation_t lhs, permutation_t rhs) const
  {
    permutation_t result;
    for (std::size_t i = 0; i < lhs.size(); i++)
    {
      result[i] = rhs[lhs[i]];
    }
    return result;
  }
};

void TestInclusiveScanWithNonCommutativeOp()
{
  const thrust::device_vector<permutation_t> input = {
    {3, 2, 0, 1, 4},
    {2, 4, 0, 1, 3},
    {3, 2, 1, 4, 0},
    {4, 3, 1, 0, 2},
    {0, 3, 2, 4, 1},
    {3, 2, 1, 0, 4},
    {3, 4, 1, 2, 0},
    {4, 2, 1, 0, 3},
    {4, 0, 1, 3, 2},
    {0, 2, 3, 1, 4}};
  thrust::device_vector<permutation_t> output(10);
  constexpr auto identity = permutation_t{0, 1, 2, 3, 4};

  thrust::inclusive_scan(input.begin(), input.end(), output.begin(), composition_op_t{});
  ASSERT_EQUAL(
    output,
    (thrust::device_vector<permutation_t>{
      {3, 2, 0, 1, 4},
      {1, 0, 2, 4, 3},
      {2, 3, 1, 0, 4},
      {1, 0, 3, 4, 2},
      {3, 0, 4, 1, 2},
      {0, 3, 4, 2, 1},
      {3, 2, 0, 1, 4},
      {0, 1, 4, 2, 3},
      {4, 0, 2, 1, 3},
      {4, 0, 3, 2, 1}}));

  thrust::exclusive_scan(input.begin(), input.end(), output.begin(), identity, composition_op_t{});
  ASSERT_EQUAL(
    output,
    (thrust::device_vector<permutation_t>{
      {0, 1, 2, 3, 4},
      {3, 2, 0, 1, 4},
      {1, 0, 2, 4, 3},
      {2, 3, 1, 0, 4},
      {1, 0, 3, 4, 2},
      {3, 0, 4, 1, 2},
      {0, 3, 4, 2, 1},
      {3, 2, 0, 1, 4},
      {0, 1, 4, 2, 3},
      {4, 0, 2, 1, 3}}));
}
DECLARE_UNITTEST(TestInclusiveScanWithNonCommutativeOp);
