#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/retag.h>
#include <thrust/reduce.h>

#include <cuda/iterator>

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
void test_reduce_simple()
{
  using T = typename Vector::value_type;

  Vector v{1, -2, 3};

  // no initializer
  REQUIRE(thrust::reduce(v.begin(), v.end()) == 2);

  // with initializer
  REQUIRE(thrust::reduce(v.begin(), v.end(), (T) 10) == 12);
}
DECLARE_VECTOR_UNITTEST(test_reduce_simple);

template <typename InputIterator>
int reduce(my_system& system, InputIterator, InputIterator)
{
  system.validate_dispatch();
  return 13;
}

TEST_CASE("TestReduceDispatchExplicit", "[reduce]")
{
  thrust::device_vector<int> vec;

  my_system sys(0); // NOLINT(misc-const-correctness)
  thrust::reduce(sys, vec.begin(), vec.end());

  REQUIRE(sys.is_valid());
}

template <typename InputIterator>
int reduce(my_tag, InputIterator, InputIterator)
{
  return 13;
}

TEST_CASE("TestReduceDispatchImplicit", "[reduce]")
{
  thrust::device_vector<int> vec;

  const int result = thrust::reduce(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.end()));

  REQUIRE(13 == result);
}

template <typename T>
struct TestReduce
{
  void operator()(const size_t n)
  {
    thrust::host_vector<T> h_data   = unittest::random_integers<T>(n);
    thrust::device_vector<T> d_data = h_data;

    T init = 13;

    T h_result = thrust::reduce(h_data.begin(), h_data.end(), init);
    T d_result = thrust::reduce(d_data.begin(), d_data.end(), init);

    REQUIRE(h_result == d_result);
  }
};
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES(TestReduce, IntegralTypes);

template <class IntVector, class FloatVector>
void test_reduce_mixed_types()
{
  // make sure we get types for default args and operators correct
  IntVector int_input{1, 2, 3, 4};

  FloatVector float_input{1.5, 2.5, 3.5, 4.5};

  // float -> int should use using plus<int> operator by default
  REQUIRE(thrust::reduce(float_input.begin(), float_input.end(), (int) 0) == 10);

  // int -> float should use using plus<float> operator by default
  REQUIRE(thrust::reduce(int_input.begin(), int_input.end(), (float) 0.5) == 10.5);
}
TEST_CASE("TestReduceMixedTypesHost", "[reduce]")
{
  test_reduce_mixed_types<thrust::host_vector<int>, thrust::host_vector<float>>();
}
TEST_CASE("TestReduceMixedTypesDevice", "[reduce]")
{
  test_reduce_mixed_types<thrust::device_vector<int>, thrust::device_vector<float>>();
}

template <typename T>
struct TestReduceWithOperator
{
  void operator()(const size_t n)
  {
    thrust::host_vector<T> h_data   = unittest::random_integers<T>(n);
    thrust::device_vector<T> d_data = h_data;

    T init = 3;

    T cpu_result = thrust::reduce(h_data.begin(), h_data.end(), init, plus_mod_10<T>());
    T gpu_result = thrust::reduce(d_data.begin(), d_data.end(), init, plus_mod_10<T>());

    REQUIRE(cpu_result == gpu_result);
  }
};
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES(TestReduceWithOperator, UnsignedIntegralTypes);

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
void test_reduce_with_indirection()
{
  // add numbers modulo 3 with external lookup table
  using T = typename Vector::value_type;

  Vector data{0, 1, 2, 1, 2, 0, 1};

  Vector table{0, 1, 2, 0, 1, 2};

  const T result = thrust::reduce(data.begin(), data.end(), T(0), plus_mod3<T>(thrust::raw_pointer_cast(&table[0])));

  REQUIRE(result == T(1));
}
DECLARE_INTEGRAL_VECTOR_UNITTEST(test_reduce_with_indirection);

template <typename T>
void test_reduce_counting_iterator()
{
  size_t const n = 15 * sizeof(T);

  REQUIRE(T(n) <= unittest::truncate_to_max_representable<T>(n));

  const thrust::counting_iterator<T, thrust::host_system_tag> h_first   = thrust::make_counting_iterator<T>(0);
  const thrust::counting_iterator<T, thrust::device_system_tag> d_first = thrust::make_counting_iterator<T>(0);

  T init = unittest::random_integer<T>();

  T h_result = thrust::reduce(h_first, h_first + n, init);
  T d_result = thrust::reduce(d_first, d_first + n, init);

  // we use ASSERT_ALMOST_EQUAL because we're testing floating point types
  ASSERT_ALMOST_EQUAL(h_result, d_result);
}
DECLARE_GENERIC_UNITTEST(test_reduce_counting_iterator);

void test_reduce_with_big_indexes_helper(int magnitude)
{
  const cuda::constant_iterator<long long> begin(1);
  const cuda::constant_iterator<long long> end = begin + (1ll << magnitude);
  REQUIRE(::cuda::std::distance(begin, end) == (1ll << magnitude));

  const long long result = thrust::reduce(thrust::device, begin, end);

  REQUIRE(result == (1ll << magnitude));
}

TEST_CASE("TestReduceWithBigIndexes", "[reduce]")
{
  test_reduce_with_big_indexes_helper(30);
#ifndef THRUST_FORCE_32_BIT_OFFSET_TYPE
  test_reduce_with_big_indexes_helper(31);
  test_reduce_with_big_indexes_helper(32);
  test_reduce_with_big_indexes_helper(33);
#endif
}
