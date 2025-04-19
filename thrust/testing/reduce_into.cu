#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/retag.h>
#include <thrust/reduce.h>

#include <limits>

#include <unittest/unittest.h>

template <typename T>
struct plus_mod_10
{
  _CCCL_HOST_DEVICE T operator()(T lhs, T rhs) const
  {
    return ((lhs % 10) + (rhs % 10)) % 10;
  }
};

template <class Vector>
void TestReduceIntoSimple()
{
  using T = typename Vector::value_type;

  Vector i{1, -2, 3};
  Vector o(1);

  // no initializer
  thrust::reduce_into(i.begin(), i.end(), o.begin());
  ASSERT_EQUAL(o[0], 2);

  // with initializer
  thrust::reduce_into(i.begin(), i.end(), o.begin(), T(10));
  ASSERT_EQUAL(o[0], 12);
}
DECLARE_VECTOR_UNITTEST(TestReduceIntoSimple);

template <typename InputIterator, typename OutputIterator>
void reduce_into(my_system& system, InputIterator, InputIterator, OutputIterator output)
{
  system.validate_dispatch();
  *output = 13;
}

void TestReduceIntoDispatchExplicit()
{
  thrust::device_vector<int> i;
  thrust::device_vector<int> o(1);

  my_system sys(0);
  thrust::reduce_into(sys, i.begin(), i.end(), o.begin());

  ASSERT_EQUAL(true, sys.is_valid());
  ASSERT_EQUAL(o[0], 13);
}
DECLARE_UNITTEST(TestReduceIntoDispatchExplicit);

template <typename InputIterator, typename OutputIterator>
void reduce_into(my_tag, InputIterator, InputIterator, OutputIterator output)
{
  *output = 13;
}

void TestReduceIntoDispatchImplicit()
{
  thrust::device_vector<int> i;
  thrust::device_vector<int> o(1);

  thrust::reduce_into(
    thrust::retag<my_tag>(i.begin()), thrust::retag<my_tag>(i.end()), thrust::retag<my_tag>(o.begin()));

  ASSERT_EQUAL(o[0], 13);
}
DECLARE_UNITTEST(TestReduceIntoDispatchImplicit);

template <typename T>
struct TestReduceInto
{
  void operator()(const size_t n)
  {
    thrust::host_vector<T> h_data(unittest::random_integers<T>(n));
    thrust::device_vector<T> d_data(h_data);
    thrust::host_vector<T> h_result(1);
    thrust::device_vector<T> d_result(1);

    T init = 13;

    thrust::reduce_into(h_data.begin(), h_data.end(), h_result.begin(), init);
    thrust::reduce_into(d_data.begin(), d_data.end(), d_result.begin(), init);

    ASSERT_EQUAL(h_result, d_result);
  }
};
VariableUnitTest<TestReduceInto, IntegralTypes> TestReduceIntoInstance;

void TestReduceIntoMixedTypesHost()
{
  // make sure we get types for default args and operators correct
  thrust::host_vector<int> int_input{1, 2, 3, 4};

  thrust::host_vector<float> float_input{1.5, 2.5, 3.5, 4.5};

  // float -> int should use using plus<int> operator by default
  thrust::host_vector<int> int_output(1);
  thrust::reduce_into(float_input.begin(), float_input.end(), int_output.begin(), int(0));
  ASSERT_EQUAL(int_output[0], 10);

  // int -> float should use using plus<float> operator by default
  thrust::host_vector<float> float_output(1);
  thrust::reduce_into(int_input.begin(), int_input.end(), float_output.begin(), float(0.5));
  ASSERT_EQUAL(float_output[0], 10.5);
}
DECLARE_UNITTEST(TestReduceIntoMixedTypesHost);
void TestReduceIntoMixedTypesDevice()
{
  // make sure we get types for default args and operators correct
  thrust::device_vector<int> int_input{1, 2, 3, 4};

  thrust::device_vector<float> float_input{1.5, 2.5, 3.5, 4.5};

  // float -> int should use using plus<int> operator by default
  thrust::device_vector<int> int_output(1);
  thrust::reduce_into(float_input.begin(), float_input.end(), int_output.begin(), int(0));
  ASSERT_EQUAL(int_output[0], 10);

  // int -> float should use using plus<float> operator by default
  thrust::device_vector<float> float_output(1);
  thrust::reduce_into(int_input.begin(), int_input.end(), float_output.begin(), float(0.5));
  ASSERT_EQUAL(float_output[0], 10.5);
}
DECLARE_UNITTEST(TestReduceIntoMixedTypesDevice);

template <typename T>
struct TestReduceIntoWithOperator
{
  void operator()(const size_t n)
  {
    thrust::host_vector<T> h_data   = unittest::random_integers<T>(n);
    thrust::device_vector<T> d_data = h_data;
    thrust::host_vector<T> h_result(1);
    thrust::device_vector<T> d_result(1);

    T init = 3;

    thrust::reduce_into(h_data.begin(), h_data.end(), h_result.begin(), init, plus_mod_10<T>());
    thrust::reduce_into(d_data.begin(), d_data.end(), d_result.begin(), init, plus_mod_10<T>());

    ASSERT_EQUAL(h_result, d_result);
  }
};
VariableUnitTest<TestReduceIntoWithOperator, UnsignedIntegralTypes> TestReduceIntoWithOperatorInstance;

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
void TestReduceIntoWithIndirection()
{
  // add numbers modulo 3 with external lookup table
  using T = typename Vector::value_type;

  Vector data{0, 1, 2, 1, 2, 0, 1};

  Vector table{0, 1, 2, 0, 1, 2};

  Vector result(1);

  thrust::reduce_into(data.begin(), data.end(), result.begin(), T(0), plus_mod3<T>(thrust::raw_pointer_cast(&table[0])));

  ASSERT_EQUAL(result[0], T(1));
}
DECLARE_INTEGRAL_VECTOR_UNITTEST(TestReduceIntoWithIndirection);

template <typename T>
void TestReduceIntoCountingIterator()
{
  size_t const n = 15 * sizeof(T);

  ASSERT_LEQUAL(T(n), unittest::truncate_to_max_representable<T>(n));

  thrust::counting_iterator<T, thrust::host_system_tag> h_first   = thrust::make_counting_iterator<T>(0);
  thrust::counting_iterator<T, thrust::device_system_tag> d_first = thrust::make_counting_iterator<T>(0);
  thrust::host_vector<T> h_result(1);
  thrust::device_vector<T> d_result(1);

  T init = unittest::random_integer<T>();

  thrust::reduce_into(h_first, h_first + n, h_result.begin(), init);
  thrust::reduce_into(d_first, d_first + n, d_result.begin(), init);

  // we use ASSERT_ALMOST_EQUAL because we're testing floating point types
  ASSERT_ALMOST_EQUAL(h_result, d_result);
}
DECLARE_GENERIC_UNITTEST(TestReduceIntoCountingIterator);
