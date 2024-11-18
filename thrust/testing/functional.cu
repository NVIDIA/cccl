#include <thrust/functional.h>
#include <thrust/transform.h>

#include <algorithm>
#include <functional>

#include <unittest/unittest.h>

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_MSVC(4244 4267) // possible loss of data

// There is a unfortunate miscompilation of the gcc-11 vectorizer leading to OOB writes
// Adding this attribute suffices that this miscompilation does not appear anymore
#if _CCCL_COMPILER(GCC, >=, 11)
#  define THRUST_DISABLE_BROKEN_GCC_VECTORIZER __attribute__((optimize("no-tree-vectorize")))
#else
#  define THRUST_DISABLE_BROKEN_GCC_VECTORIZER
#endif

const size_t NUM_SAMPLES = 10000;

template <class InputVector, class OutputVector, class Operator, class ReferenceOperator>
THRUST_DISABLE_BROKEN_GCC_VECTORIZER void TestUnaryFunctional()
{
  using InputType  = typename InputVector::value_type;
  using OutputType = typename OutputVector::value_type;

  thrust::host_vector<InputType> std_input = unittest::random_samples<InputType>(NUM_SAMPLES);
  thrust::host_vector<OutputType> std_output(NUM_SAMPLES);

  InputVector input = std_input;
  OutputVector output(NUM_SAMPLES);

  thrust::transform(input.begin(), input.end(), output.begin(), Operator());
  thrust::transform(std_input.begin(), std_input.end(), std_output.begin(), ReferenceOperator());

  ASSERT_EQUAL(output, std_output);
}

template <class InputVector, class OutputVector, class Operator, class ReferenceOperator>
THRUST_DISABLE_BROKEN_GCC_VECTORIZER void TestBinaryFunctional()
{
  using InputType  = typename InputVector::value_type;
  using OutputType = typename OutputVector::value_type;

  thrust::host_vector<InputType> std_input1 = unittest::random_samples<InputType>(NUM_SAMPLES);
  thrust::host_vector<InputType> std_input2 = unittest::random_samples<InputType>(NUM_SAMPLES);
  thrust::host_vector<OutputType> std_output(NUM_SAMPLES);

  // Replace zeros to avoid divide by zero exceptions
  std::replace(std_input2.begin(), std_input2.end(), (InputType) 0, (InputType) 1);

  InputVector input1 = std_input1;
  InputVector input2 = std_input2;
  OutputVector output(NUM_SAMPLES);

  thrust::transform(input1.begin(), input1.end(), input2.begin(), output.begin(), Operator());
  thrust::transform(std_input1.begin(), std_input1.end(), std_input2.begin(), std_output.begin(), ReferenceOperator());

  // Note: FP division is not bit-equal, even when nvcc is invoked with --prec-div
  ASSERT_ALMOST_EQUAL(output, std_output);
}

// XXX add bool to list
// Instantiate a macro for all integer-like data types
// clang-format off
#define INSTANTIATE_INTEGER_TYPES(Macro, vector_type, operator_name)   \
Macro(vector_type, operator_name, unittest::int8_t  )                  \
Macro(vector_type, operator_name, unittest::uint8_t )                  \
Macro(vector_type, operator_name, unittest::int16_t )                  \
Macro(vector_type, operator_name, unittest::uint16_t)                  \
Macro(vector_type, operator_name, unittest::int32_t )                  \
Macro(vector_type, operator_name, unittest::uint32_t)                  \
Macro(vector_type, operator_name, unittest::int64_t )                  \
Macro(vector_type, operator_name, unittest::uint64_t)
// clang-format on

// Instantiate a macro for all integer and floating point data types
#define INSTANTIATE_ALL_TYPES(Macro, vector_type, operator_name) \
  INSTANTIATE_INTEGER_TYPES(Macro, vector_type, operator_name)   \
  Macro(vector_type, operator_name, float)

// op(T) -> T
#define INSTANTIATE_UNARY_ARITHMETIC_FUNCTIONAL_TEST(vector_type, operator_name, data_type) \
  TestUnaryFunctional<thrust::vector_type<data_type>,                                       \
                      thrust::vector_type<data_type>,                                       \
                      thrust::operator_name<data_type>,                                     \
                      std::operator_name<data_type>>();
// XXX revert OutputVector<T> back to bool
// op(T) -> bool
#define INSTANTIATE_UNARY_LOGICAL_FUNCTIONAL_TEST(vector_type, operator_name, data_type) \
  TestUnaryFunctional<thrust::vector_type<data_type>,                                    \
                      thrust::vector_type<data_type>,                                    \
                      thrust::operator_name<data_type>,                                  \
                      std::operator_name<data_type>>();
// op(T,T) -> T
#define INSTANTIATE_BINARY_ARITHMETIC_FUNCTIONAL_TEST(vector_type, operator_name, data_type) \
  TestBinaryFunctional<thrust::vector_type<data_type>,                                       \
                       thrust::vector_type<data_type>,                                       \
                       thrust::operator_name<data_type>,                                     \
                       std::operator_name<data_type>>();
// XXX revert OutputVector<T> back to bool
// op(T,T) -> bool
#define INSTANTIATE_BINARY_LOGICAL_FUNCTIONAL_TEST(vector_type, operator_name, data_type) \
  TestBinaryFunctional<thrust::vector_type<data_type>,                                    \
                       thrust::vector_type<data_type>,                                    \
                       thrust::operator_name<data_type>,                                  \
                       std::operator_name<data_type>>();

// op(T) -> T
#define DECLARE_UNARY_ARITHMETIC_FUNCTIONAL_UNITTEST(operator_name, OperatorName)                      \
  void Test##OperatorName##FunctionalHost()                                                            \
  {                                                                                                    \
    INSTANTIATE_ALL_TYPES(INSTANTIATE_UNARY_ARITHMETIC_FUNCTIONAL_TEST, host_vector, operator_name);   \
  }                                                                                                    \
  DECLARE_UNITTEST(Test##OperatorName##FunctionalHost);                                                \
  void Test##OperatorName##FunctionalDevice()                                                          \
  {                                                                                                    \
    INSTANTIATE_ALL_TYPES(INSTANTIATE_UNARY_ARITHMETIC_FUNCTIONAL_TEST, device_vector, operator_name); \
  }                                                                                                    \
  DECLARE_UNITTEST(Test##OperatorName##FunctionalDevice);

// op(T) -> bool
#define DECLARE_UNARY_LOGICAL_FUNCTIONAL_UNITTEST(operator_name, OperatorName)                      \
  void Test##OperatorName##FunctionalHost()                                                         \
  {                                                                                                 \
    INSTANTIATE_ALL_TYPES(INSTANTIATE_UNARY_LOGICAL_FUNCTIONAL_TEST, host_vector, operator_name);   \
  }                                                                                                 \
  DECLARE_UNITTEST(Test##OperatorName##FunctionalHost);                                             \
  void Test##OperatorName##FunctionalDevice()                                                       \
  {                                                                                                 \
    INSTANTIATE_ALL_TYPES(INSTANTIATE_UNARY_LOGICAL_FUNCTIONAL_TEST, device_vector, operator_name); \
  }                                                                                                 \
  DECLARE_UNITTEST(Test##OperatorName##FunctionalDevice);

// op(T,T) -> T
#define DECLARE_BINARY_ARITHMETIC_FUNCTIONAL_UNITTEST(operator_name, OperatorName)                      \
  void Test##OperatorName##FunctionalHost()                                                             \
  {                                                                                                     \
    INSTANTIATE_ALL_TYPES(INSTANTIATE_BINARY_ARITHMETIC_FUNCTIONAL_TEST, host_vector, operator_name);   \
  }                                                                                                     \
  DECLARE_UNITTEST(Test##OperatorName##FunctionalHost);                                                 \
  void Test##OperatorName##FunctionalDevice()                                                           \
  {                                                                                                     \
    INSTANTIATE_ALL_TYPES(INSTANTIATE_BINARY_ARITHMETIC_FUNCTIONAL_TEST, device_vector, operator_name); \
  }                                                                                                     \
  DECLARE_UNITTEST(Test##OperatorName##FunctionalDevice);

// op(T,T) -> T (for integer T only)
#define DECLARE_BINARY_INTEGER_ARITHMETIC_FUNCTIONAL_UNITTEST(operator_name, OperatorName)                  \
  void Test##OperatorName##FunctionalHost()                                                                 \
  {                                                                                                         \
    INSTANTIATE_INTEGER_TYPES(INSTANTIATE_BINARY_ARITHMETIC_FUNCTIONAL_TEST, host_vector, operator_name);   \
  }                                                                                                         \
  DECLARE_UNITTEST(Test##OperatorName##FunctionalHost);                                                     \
  void Test##OperatorName##FunctionalDevice()                                                               \
  {                                                                                                         \
    INSTANTIATE_INTEGER_TYPES(INSTANTIATE_BINARY_ARITHMETIC_FUNCTIONAL_TEST, device_vector, operator_name); \
  }                                                                                                         \
  DECLARE_UNITTEST(Test##OperatorName##FunctionalDevice);

// op(T,T) -> bool
#define DECLARE_BINARY_LOGICAL_FUNCTIONAL_UNITTEST(operator_name, OperatorName)                      \
  void Test##OperatorName##FunctionalHost()                                                          \
  {                                                                                                  \
    INSTANTIATE_ALL_TYPES(INSTANTIATE_BINARY_LOGICAL_FUNCTIONAL_TEST, host_vector, operator_name);   \
  }                                                                                                  \
  DECLARE_UNITTEST(Test##OperatorName##FunctionalHost);                                              \
  void Test##OperatorName##FunctionalDevice()                                                        \
  {                                                                                                  \
    INSTANTIATE_ALL_TYPES(INSTANTIATE_BINARY_LOGICAL_FUNCTIONAL_TEST, device_vector, operator_name); \
  }                                                                                                  \
  DECLARE_UNITTEST(Test##OperatorName##FunctionalDevice);

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_MSVC(4146) // warning C4146: unary minus operator applied to unsigned type, result still unsigned

// Create the unit tests
DECLARE_UNARY_ARITHMETIC_FUNCTIONAL_UNITTEST(negate, Negate);
_CCCL_DIAG_POP
DECLARE_UNARY_LOGICAL_FUNCTIONAL_UNITTEST(logical_not, LogicalNot);

// TODO(bgruber): replace by cuda::std::as_const in C++14
template <class _Tp>
typename ::cuda::std::add_const<_Tp>::type& as_const(_Tp& __t) noexcept
{
  return __t;
}

// Ad-hoc testing for other functionals
THRUST_DISABLE_BROKEN_GCC_VECTORIZER void TestIdentityFunctional()
{
  int i    = 42;
  double d = 3.14;

  // pass through
  ASSERT_EQUAL(thrust::identity<int>{}(i), 42);
  ASSERT_EQUAL(thrust::identity<int>{}(d), 3);

  // modification through
  thrust::identity<int>{}(i) = 1337;
  ASSERT_EQUAL(i, 1337);

  // value categories and const
  static_assert(::cuda::std::is_same<decltype(thrust::identity<int>{}(42)), int&&>::value, "");
  static_assert(::cuda::std::is_same<decltype(thrust::identity<int>{}(i)), int&>::value, "");
  static_assert(::cuda::std::is_same<decltype(thrust::identity<int>{}(as_const(i))), const int&>::value, "");
  static_assert(::cuda::std::is_same<decltype(thrust::identity<int>{}(::cuda::std::move(i))), int&&>::value, "");
  static_assert(::cuda::std::is_same<decltype(thrust::identity<int>{}(static_cast<const int&&>(i))), const int&>::value,
                "");

  // value categories when casting to different type
  static_assert(::cuda::std::is_same<decltype(thrust::identity<int>{}(3.14)), int&&>::value, "");
  // unfortunately, old versions of MSVC pick the `const int&` overload instead of `int&&`
#if defined(_CCCL_COMPILER_MSVC) && _CCCL_MSVC_VERSION >= 1929
  static_assert(::cuda::std::is_same<decltype(thrust::identity<int>{}(d)), int&&>::value, "");
  static_assert(::cuda::std::is_same<decltype(thrust::identity<int>{}(as_const(d))), int&&>::value, "");
#endif
  static_assert(::cuda::std::is_same<decltype(thrust::identity<int>{}(::cuda::std::move(d))), int&&>::value, "");
  static_assert(::cuda::std::is_same<decltype(thrust::identity<int>{}(static_cast<const double&&>(d))), int&&>::value,
                "");
}
DECLARE_UNITTEST(TestIdentityFunctional);

template <class Vector>
THRUST_DISABLE_BROKEN_GCC_VECTORIZER void TestIdentityFunctionalVector()
{
  using T = typename Vector::value_type;
  Vector input{0, 1, 2, 3};
  Vector output(4);
  thrust::transform(input.begin(), input.end(), output.begin(), thrust::identity<T>());
  ASSERT_EQUAL(input, output);
}
DECLARE_VECTOR_UNITTEST(TestIdentityFunctionalVector);

template <class Vector>
THRUST_DISABLE_BROKEN_GCC_VECTORIZER void TestProject1stFunctional()
{
  using T = typename Vector::value_type;

  Vector lhs{0, 1, 2, 3};
  Vector rhs{3, 4, 5, 6};

  Vector output(4);

  thrust::transform(lhs.begin(), lhs.end(), rhs.begin(), output.begin(), thrust::project1st<T, T>());

  ASSERT_EQUAL(output, lhs);
}
DECLARE_VECTOR_UNITTEST(TestProject1stFunctional);

template <class Vector>
THRUST_DISABLE_BROKEN_GCC_VECTORIZER void TestProject2ndFunctional()
{
  using T = typename Vector::value_type;

  Vector lhs{0, 1, 2, 3};
  Vector rhs{3, 4, 5, 6};

  Vector output(4);

  thrust::transform(lhs.begin(), lhs.end(), rhs.begin(), output.begin(), thrust::project2nd<T, T>());

  ASSERT_EQUAL(output, rhs);
}
DECLARE_VECTOR_UNITTEST(TestProject2ndFunctional);

template <class Vector>
THRUST_DISABLE_BROKEN_GCC_VECTORIZER void TestMaximumFunctional()
{
  using T = typename Vector::value_type;

  Vector input1{8, 3, 7, 7};
  Vector input2{5, 6, 9, 3};

  Vector output(4);

  thrust::transform(input1.begin(), input1.end(), input2.begin(), output.begin(), thrust::maximum<T>());

  Vector ref{8, 6, 9, 7};
  ASSERT_EQUAL(output, ref);
}
DECLARE_VECTOR_UNITTEST(TestMaximumFunctional);

template <class Vector>
THRUST_DISABLE_BROKEN_GCC_VECTORIZER void TestMinimumFunctional()
{
  using T = typename Vector::value_type;

  Vector input1{8, 3, 7, 7};
  Vector input2{5, 6, 9, 3};

  Vector output(4);

  thrust::transform(input1.begin(), input1.end(), input2.begin(), output.begin(), thrust::minimum<T>());

  Vector ref{5, 3, 7, 3};
  ASSERT_EQUAL(output, ref);
}
DECLARE_VECTOR_UNITTEST(TestMinimumFunctional);

template <class Vector>
THRUST_DISABLE_BROKEN_GCC_VECTORIZER void TestNot1()
{
  using T = typename Vector::value_type;

  Vector input{1, 0, 1, 1, 0};

  Vector output(5);

  thrust::transform(input.begin(), input.end(), output.begin(), thrust::not_fn(thrust::identity<T>()));

  Vector ref{0, 1, 0, 0, 1};
  ASSERT_EQUAL(output, ref);
}
DECLARE_INTEGRAL_VECTOR_UNITTEST(TestNot1);

// GCC 11 fails to build this test case with a spurious error in a
// very specific scenario:
// - GCC 11
// - CPP system for both host and device
// - C++11 dialect
#if !(_CCCL_COMPILER(GCC, >=, 11) && _CCCL_COMPILER(GCC, <, 12) && THRUST_HOST_SYSTEM == THRUST_HOST_SYSTEM_CPP \
      && THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CPP && _CCCL_STD_VER == 2011)

template <class Vector>
THRUST_DISABLE_BROKEN_GCC_VECTORIZER void TestNot2()
{
  using T = typename Vector::value_type;

  Vector input1{1, 0, 1, 1, 0};
  Vector input2{1, 1, 0, 1, 1};

  Vector output(5);

  thrust::transform(input1.begin(), input1.end(), input2.begin(), output.begin(), thrust::not_fn(thrust::equal_to<T>()));

  Vector ref{0, 1, 1, 0, 1};
  ASSERT_EQUAL(output, ref);
}
DECLARE_VECTOR_UNITTEST(TestNot2);

#endif // Weird GCC11 failure case

_CCCL_DIAG_POP
