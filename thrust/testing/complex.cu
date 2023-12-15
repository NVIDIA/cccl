#include <thrust/complex.h>
#include <thrust/detail/config.h>

#include <complex>
#include <iostream>
#include <sstream>

#include <unittest/assertions.h>
#include <unittest/unittest.h>

/*
   The following tests do not check for the numerical accuracy of the operations.
   That is tested in a separate program (complex_accuracy.cpp) which requires mpfr,
   and takes a lot of time to run.
 */

namespace
{

// Helper construction to create a double from float and
// vice versa to test thrust::complex promoting operators.
template <typename T>
struct other_floating_point_type
{};

template <>
struct other_floating_point_type<float>
{
  using type = double;
};

template <>
struct other_floating_point_type<double>
{
  using type = float;
};

template <typename T>
using other_floating_point_type_t = typename other_floating_point_type<T>::type;

} // anonymous namespace

template <typename T>
struct TestComplexSizeAndAlignment
{
  void operator()()
  {
    THRUST_STATIC_ASSERT(sizeof(thrust::complex<T>) == sizeof(T) * 2);
    THRUST_STATIC_ASSERT(THRUST_ALIGNOF(thrust::complex<T>) == THRUST_ALIGNOF(T) * 2);

    THRUST_STATIC_ASSERT(sizeof(thrust::complex<T const>) == sizeof(T) * 2);
    THRUST_STATIC_ASSERT(THRUST_ALIGNOF(thrust::complex<T const>) == THRUST_ALIGNOF(T) * 2);
  }
};
SimpleUnitTest<TestComplexSizeAndAlignment, FloatingPointTypes> TestComplexSizeAndAlignmentInstance;

template <typename T>
struct TestComplexTypeCheck
{
  void operator()()
  {
    THRUST_STATIC_ASSERT(thrust::is_complex<thrust::complex<T>>::value);
    THRUST_STATIC_ASSERT(thrust::is_complex<std::complex<T>>::value);
    THRUST_STATIC_ASSERT(thrust::is_complex<cuda::std::complex<T>>::value);
  }
};
SimpleUnitTest<TestComplexTypeCheck, FloatingPointTypes> TestComplexTypeCheckInstance;

template <typename T>
struct TestComplexConstructionAndAssignment
{
  void operator()(void)
  {
    thrust::host_vector<T> data = unittest::random_samples<T>(2);

    const T real = data[0];
    const T imag = data[1];

    {
      const thrust::complex<T> construct_from_real_and_imag(real, imag);
      ASSERT_EQUAL(real, construct_from_real_and_imag.real());
      ASSERT_EQUAL(imag, construct_from_real_and_imag.imag());
    }

    {
      const thrust::complex<T> construct_from_real(real);
      ASSERT_EQUAL(real, construct_from_real.real());
      ASSERT_EQUAL(T(), construct_from_real.imag());
    }

    {
      const thrust::complex<T> expected(real, imag);
      thrust::complex<T> construct_from_copy(expected);
      ASSERT_EQUAL(expected.real(), construct_from_copy.real());
      ASSERT_EQUAL(expected.imag(), construct_from_copy.imag());
    }

    {
      thrust::complex<T> construct_from_move(thrust::complex<T>(real, imag));
      ASSERT_EQUAL(real, construct_from_move.real());
      ASSERT_EQUAL(imag, construct_from_move.imag());
    }

    {
      thrust::complex<T> copy_assign{};
      const thrust::complex<T> expected{real, imag};
      copy_assign = expected;
      ASSERT_EQUAL(expected.real(), copy_assign.real());
      ASSERT_EQUAL(expected.imag(), copy_assign.imag());
    }

    {
      thrust::complex<T> move_assign{};
      const thrust::complex<T> expected{real, imag};
      move_assign = thrust::complex<T>{real, imag};
      ASSERT_EQUAL(expected.real(), move_assign.real());
      ASSERT_EQUAL(expected.imag(), move_assign.imag());
    }

    {
      thrust::complex<T> assign_from_lvalue_T{};
      const thrust::complex<T> expected{real};
      const T to_be_copied = real;
      assign_from_lvalue_T = to_be_copied;
      ASSERT_EQUAL(expected.real(), assign_from_lvalue_T.real());
      ASSERT_EQUAL(expected.imag(), assign_from_lvalue_T.imag());
    }

    {
      thrust::complex<T> assign_from_rvalue_T{};
      const thrust::complex<T> expected{real};
      assign_from_rvalue_T = T(real);
      ASSERT_EQUAL(expected.real(), assign_from_rvalue_T.real());
      ASSERT_EQUAL(expected.imag(), assign_from_rvalue_T.imag());
    }

    {
      const std::complex<T> expected(real, imag);
      const thrust::complex<T> copy_from_std(expected);
      ASSERT_EQUAL(expected.real(), copy_from_std.real());
      ASSERT_EQUAL(expected.imag(), copy_from_std.imag());
    }

    {
      thrust::complex<T> assign_from_lvalue_std{};
      const std::complex<T> expected(real, imag);
      assign_from_lvalue_std = expected;
      ASSERT_EQUAL(expected.real(), assign_from_lvalue_std.real());
      ASSERT_EQUAL(expected.imag(), assign_from_lvalue_std.imag());
    }

    {
      thrust::complex<T> assign_from_rvalue_std{};
      assign_from_rvalue_std = std::complex<T>(real, imag);
      ASSERT_EQUAL(real, assign_from_rvalue_std.real());
      ASSERT_EQUAL(imag, assign_from_rvalue_std.imag());
    }
  }
};
SimpleUnitTest<TestComplexConstructionAndAssignment, FloatingPointTypes>
  TestComplexConstructionAndAssignmentInstance;

template <typename T>
struct TestComplexConstructionAndAssignmentWithPromoting
{
  void operator()(void)
  {
    using T0 = T;
    using T1 = other_floating_point_type_t<T0>;

    thrust::host_vector<T0> data = unittest::random_samples<T0>(2);

    const T0 real_T0 = data[0];
    const T0 imag_T0 = data[1];

    const T1 real_T1 = static_cast<T1>(real_T0);
    const T1 imag_T1 = static_cast<T1>(imag_T0);

    {
      const thrust::complex<T0> construct_from_real_and_imag(real_T1, imag_T1);
      ASSERT_ALMOST_EQUAL(real_T0, construct_from_real_and_imag.real());
      ASSERT_ALMOST_EQUAL(imag_T0, construct_from_real_and_imag.imag());
    }

    {
      const thrust::complex<T0> construct_from_real(real_T1);
      ASSERT_ALMOST_EQUAL(real_T0, construct_from_real.real());
      ASSERT_EQUAL(T0(), construct_from_real.imag());
    }

    {
      const thrust::complex<T1> expected(real_T1, imag_T1);
      thrust::complex<T0> construct_from_copy(expected);
      ASSERT_ALMOST_EQUAL(real_T0, construct_from_copy.real());
      ASSERT_ALMOST_EQUAL(imag_T0, construct_from_copy.imag());
    }

    {
      thrust::complex<T0> construct_from_move(thrust::complex<T1>(real_T1, imag_T1));
      ASSERT_ALMOST_EQUAL(real_T0, construct_from_move.real());
      ASSERT_ALMOST_EQUAL(imag_T0, construct_from_move.imag());
    }

    {
      thrust::complex<T0> copy_assign{};
      const thrust::complex<T1> expected{real_T1, imag_T1};
      copy_assign = expected;
      ASSERT_EQUAL(expected.real(), copy_assign.real());
      ASSERT_EQUAL(expected.imag(), copy_assign.imag());
    }

    {
      thrust::complex<T0> assign_from_lvalue_T{};
      const thrust::complex<T1> expected{real_T1};
      const T1 to_be_copied = real_T1;
      assign_from_lvalue_T  = to_be_copied;
      ASSERT_ALMOST_EQUAL(expected.real(), assign_from_lvalue_T.real());
      ASSERT_EQUAL(expected.imag(), assign_from_lvalue_T.imag());
    }

    {
      const std::complex<T1> expected(real_T1, imag_T1);
      const thrust::complex<T0> copy_from_std(expected);
      ASSERT_ALMOST_EQUAL(expected.real(), copy_from_std.real());
      ASSERT_ALMOST_EQUAL(expected.imag(), copy_from_std.imag());
    }

    {
      thrust::complex<T1> assign_from_lvalue_std{};
      const std::complex<T0> expected(real_T1, imag_T1);
      assign_from_lvalue_std = expected;
      ASSERT_ALMOST_EQUAL(expected.real(), assign_from_lvalue_std.real());
      ASSERT_ALMOST_EQUAL(expected.imag(), assign_from_lvalue_std.imag());
    }

    {
      thrust::complex<T0> assign_from_rvalue_std{};
      assign_from_rvalue_std = std::complex<T1>(real_T1, imag_T1);
      ASSERT_ALMOST_EQUAL(real_T0, assign_from_rvalue_std.real());
      ASSERT_ALMOST_EQUAL(imag_T1, assign_from_rvalue_std.imag());
    }
  }
};
SimpleUnitTest<TestComplexConstructionAndAssignmentWithPromoting, FloatingPointTypes>
  TestComplexConstructionAndAssignmentWithPromotingInstance;

template <typename T>
struct TestComplexGetters
{
  void operator()(void)
  {
    thrust::host_vector<T> data = unittest::random_samples<T>(2);

    thrust::complex<T> z(data[0], data[1]);

    ASSERT_EQUAL(data[0], z.real());
    ASSERT_EQUAL(data[1], z.imag());

    z.real(data[1]);
    z.imag(data[0]);
    ASSERT_EQUAL(data[1], z.real());
    ASSERT_EQUAL(data[0], z.imag());

    volatile thrust::complex<T> v(data[0], data[1]);

    ASSERT_EQUAL(data[0], v.real());
    ASSERT_EQUAL(data[1], v.imag());

    v.real(data[1]);
    v.imag(data[0]);
    ASSERT_EQUAL(data[1], v.real());
    ASSERT_EQUAL(data[0], v.imag());
  }
};
SimpleUnitTest<TestComplexGetters, FloatingPointTypes> TestComplexGettersInstance;

template <typename T>
struct TestComplexComparisionOperators
{
  void operator()(void)
  {
    {
      thrust::host_vector<T> data = unittest::random_samples<T>(1);

      const T a = data[0];
      const T b = data[0] + T(1.0);

      ASSERT_EQUAL(thrust::complex<T>(a, b) == thrust::complex<T>(a, b), true);
      ASSERT_EQUAL(thrust::complex<T>(a, T()) == a, true);
      ASSERT_EQUAL(a == thrust::complex<T>(a, T()), true);

      ASSERT_EQUAL(thrust::complex<T>(a, b) == std::complex<T>(a, b), true);
      ASSERT_EQUAL(std::complex<T>(a, b) == thrust::complex<T>(a, b), true);

      ASSERT_EQUAL(thrust::complex<T>(a, b) != thrust::complex<T>(b, a), true);
      ASSERT_EQUAL(thrust::complex<T>(a, T()) != b, true);
      ASSERT_EQUAL(b != thrust::complex<T>(a, T()), true);

      ASSERT_EQUAL(thrust::complex<T>(a, b) != a, true);
      ASSERT_EQUAL(a != thrust::complex<T>(a, b), true);

      ASSERT_EQUAL(thrust::complex<T>(a, b) != std::complex<T>(b, a), true);
      ASSERT_EQUAL(std::complex<T>(a, b) != thrust::complex<T>(b, a), true);
    }

    // Testing comparison operators with promoted types.
    // These tests don't use random numbers on purpose since `T0(x) == T1(x)` will
    // not be true for all x.
    {
      using T0 = T;
      using T1 = other_floating_point_type_t<T0>;

      ASSERT_EQUAL(thrust::complex<T0>(1.0, 2.0) == thrust::complex<T1>(1.0, 2.0), true);
      ASSERT_EQUAL(thrust::complex<T0>(1.0, T0()) == T1(1.0), true);
      ASSERT_EQUAL(T1(1.0) == thrust::complex<T0>(1.0, 0.0), true);

      ASSERT_EQUAL(thrust::complex<T0>(1.0, 2.0) == std::complex<T1>(1.0, 2.0), true);
      ASSERT_EQUAL(std::complex<T0>(1.0, 2.0) == thrust::complex<T1>(1.0, 2.0), true);

      ASSERT_EQUAL(thrust::complex<T0>(1.0, 2.0) != T1(1.0), true);
      ASSERT_EQUAL(T1(1.0) != thrust::complex<T0>(1.0, 2.0), true);
    }
  }
};
SimpleUnitTest<TestComplexComparisionOperators, FloatingPointTypes>
  TestComplexComparisionOperatorsInstance;

template <typename T>
struct TestComplexMemberOperators
{
  void operator()(void)
  {
    {
      thrust::host_vector<T> data = unittest::random_samples<T>(5);

      thrust::complex<T> a_thrust(data[0], data[1]);
      const thrust::complex<T> b_thrust(data[2], data[3]);

      std::complex<T> a_std(a_thrust);
      const std::complex<T> b_std(b_thrust);

      a_thrust += b_thrust;
      a_std += b_std;
      ASSERT_ALMOST_EQUAL(a_thrust, a_std);

      a_thrust -= b_thrust;
      a_std -= b_std;
      ASSERT_ALMOST_EQUAL(a_thrust, a_std);

      a_thrust *= b_thrust;
      a_std *= b_std;
      ASSERT_ALMOST_EQUAL(a_thrust, a_std);

      a_thrust /= b_thrust;
      a_std /= b_std;
      ASSERT_ALMOST_EQUAL(a_thrust, a_std);

      // arithmetic operators with `double` and `float`
      const T real = data[4];

      a_thrust += real;
      a_std += std::complex<T>(real);
      ASSERT_ALMOST_EQUAL(a_thrust, a_std);

      a_thrust -= real;
      a_std -= std::complex<T>(real);
      ASSERT_ALMOST_EQUAL(a_thrust, a_std);

      a_thrust *= real;
      a_std *= std::complex<T>(real);
      ASSERT_ALMOST_EQUAL(a_thrust, a_std);

      a_thrust /= real;
      a_std /= std::complex<T>(real);
      ASSERT_ALMOST_EQUAL(a_thrust, a_std);

      // casting operator
      a_std = (std::complex<T>)a_thrust;
      ASSERT_ALMOST_EQUAL(a_thrust.real(), a_std.real());
      ASSERT_ALMOST_EQUAL(a_thrust.imag(), a_std.imag());
    }

    // Testing arithmetic member operators with promoted types.
    {
      using T0 = T;
      using T1 = other_floating_point_type_t<T0>;

      thrust::host_vector<T0> data = unittest::random_samples<T0>(5);

      thrust::complex<T0> a_thrust(data[0], data[1]);
      const thrust::complex<T1> b_thrust(data[2], data[3]);

      std::complex<T1> a_std(data[0], data[1]);
      const std::complex<T1> b_std(data[2], data[3]);

      // The following tests require that thrust::complex and std::complex are `almost` equal.
      ASSERT_ALMOST_EQUAL(a_thrust, a_std);
      ASSERT_ALMOST_EQUAL(b_thrust, b_thrust);

      a_thrust += b_thrust;
      a_std += b_std;
      ASSERT_ALMOST_EQUAL(a_thrust, a_std);

      a_thrust -= b_thrust;
      a_std -= b_std;
      ASSERT_ALMOST_EQUAL(a_thrust, a_std);

      a_thrust *= b_thrust;
      a_std *= b_std;
      ASSERT_ALMOST_EQUAL(a_thrust, a_std);

      a_thrust /= b_thrust;
      a_std /= b_std;
      ASSERT_ALMOST_EQUAL(a_thrust, a_std);

      // Testing arithmetic member operators with another floating point type.
      const T1 e = data[2];

      a_thrust += e;
      a_std += std::complex<T1>(e);
      ASSERT_ALMOST_EQUAL(a_thrust, a_std);

      a_thrust -= e;
      a_std -= std::complex<T1>(e);
      ASSERT_ALMOST_EQUAL(a_thrust, a_std);

      a_thrust *= e;
      a_std *= std::complex<T1>(e);
      ASSERT_ALMOST_EQUAL(a_thrust, a_std);

      a_thrust /= e;
      a_std /= std::complex<T1>(e);
      ASSERT_ALMOST_EQUAL(a_thrust, a_std);
    }
  }
};
SimpleUnitTest<TestComplexMemberOperators, FloatingPointTypes> TestComplexMemberOperatorsInstance;

template <typename T>
struct TestComplexBasicArithmetic
{
  void operator()(void)
  {
    thrust::host_vector<T> data = unittest::random_samples<T>(2);

    const thrust::complex<T> a(data[0], data[1]);
    const std::complex<T> b(a);

    // Test the basic arithmetic functions against std

    ASSERT_ALMOST_EQUAL(thrust::abs(a), std::abs(b));
    ASSERT_ALMOST_EQUAL(thrust::arg(a), std::arg(b));
    ASSERT_ALMOST_EQUAL(thrust::norm(a), std::norm(b));

    ASSERT_EQUAL(thrust::conj(a), std::conj(b));
    static_assert(cuda::std::is_same<thrust::complex<T>, decltype(thrust::conj(a))>::value, "");

    ASSERT_ALMOST_EQUAL(thrust::polar(data[0], data[1]), std::polar(data[0], data[1]));
    static_assert(cuda::std::is_same<thrust::complex<T>, decltype(thrust::polar(data[0], data[1]))>::value, "");

    // random_samples does not seem to produce infinities so proj(z) == z
    ASSERT_EQUAL(thrust::proj(a), a);
    static_assert(cuda::std::is_same<thrust::complex<T>, decltype(thrust::proj(a))>::value, "");
  }
};
SimpleUnitTest<TestComplexBasicArithmetic, FloatingPointTypes> TestComplexBasicArithmeticInstance;

template <typename T>
struct TestComplexBinaryArithmetic
{
  void operator()(void)
  {
    {
      thrust::host_vector<T> data = unittest::random_samples<T>(5);

      const thrust::complex<T> a(data[0], data[1]);
      const thrust::complex<T> b(data[2], data[3]);
      const T real = data[4];

      ASSERT_ALMOST_EQUAL(a * b, std::complex<T>(a) * std::complex<T>(b));
      ASSERT_ALMOST_EQUAL(a * real, std::complex<T>(a) * real);
      ASSERT_ALMOST_EQUAL(real * b, real * std::complex<T>(b));

      ASSERT_ALMOST_EQUAL(a / b, std::complex<T>(a) / std::complex<T>(b));
      ASSERT_ALMOST_EQUAL(a / real, std::complex<T>(a) / real);
      ASSERT_ALMOST_EQUAL(real / b, real / std::complex<T>(b));

      ASSERT_ALMOST_EQUAL(a + b, std::complex<T>(a) + std::complex<T>(b));
      ASSERT_ALMOST_EQUAL(a + real, std::complex<T>(a) + real);
      ASSERT_ALMOST_EQUAL(real + b, real + std::complex<T>(b));

      ASSERT_ALMOST_EQUAL(a - b, std::complex<T>(a) - std::complex<T>(b));
      ASSERT_ALMOST_EQUAL(a - real, std::complex<T>(a) - real);
      ASSERT_ALMOST_EQUAL(real - b, real - std::complex<T>(b));
    }

    // Testing binary arithmetic with promoted types.
    {
      using T0 = T;
      using T1 = other_floating_point_type_t<T0>;

      thrust::host_vector<T0> data = unittest::random_samples<T0>(5);

      const thrust::complex<T0> a_thrust(data[0], data[1]);
      const thrust::complex<T1> b_thrust(data[2], data[3]);
      const thrust::complex<T0> a_std(data[0], data[1]);
      const thrust::complex<T0> b_std(data[2], data[3]);

      const T0 real_T0 = data[4];
      const T1 real_T1 = static_cast<T1>(real_T0);

      ASSERT_ALMOST_EQUAL(a_thrust * b_thrust, a_std * b_std);
      ASSERT_ALMOST_EQUAL(a_thrust * real_T1, a_std * real_T0);
      ASSERT_ALMOST_EQUAL(real_T0 * b_thrust, real_T0 * b_std);

      ASSERT_ALMOST_EQUAL(a_thrust / b_thrust, a_std / b_std);
      ASSERT_ALMOST_EQUAL(a_thrust / real_T1, a_std / real_T0);
      ASSERT_ALMOST_EQUAL(real_T0 / b_thrust, real_T0 / b_std);

      ASSERT_ALMOST_EQUAL(a_thrust + b_thrust, a_std + b_std);
      ASSERT_ALMOST_EQUAL(a_thrust + real_T1, a_std + real_T0);
      ASSERT_ALMOST_EQUAL(real_T0 + b_thrust, real_T0 + b_std);

      ASSERT_ALMOST_EQUAL(a_thrust - b_thrust, a_std - b_std);
      ASSERT_ALMOST_EQUAL(a_thrust - real_T1, a_std - real_T0);
      ASSERT_ALMOST_EQUAL(real_T0 - b_thrust, real_T0 - b_std);
    }
  }
};
SimpleUnitTest<TestComplexBinaryArithmetic, FloatingPointTypes> TestComplexBinaryArithmeticInstance;

template <typename T>
struct TestComplexUnaryArithmetic
{
  void operator()(void)
  {
    thrust::host_vector<T> data = unittest::random_samples<T>(2);

    const thrust::complex<T> a(data[0], data[1]);

    ASSERT_EQUAL(+a, a);
    ASSERT_EQUAL(-a, a * (-1.0));
  }
};
SimpleUnitTest<TestComplexUnaryArithmetic, FloatingPointTypes> TestComplexUnaryArithmeticInstance;

template <typename T>
struct TestComplexExponentialFunctions
{
  void operator()(void)
  {
    thrust::host_vector<T> data = unittest::random_samples<T>(2);

    const thrust::complex<T> a(data[0], data[1]);
    const std::complex<T> b(a);

    ASSERT_ALMOST_EQUAL(thrust::exp(a), std::exp(b));
    ASSERT_ALMOST_EQUAL(thrust::log(a), std::log(b));
    ASSERT_ALMOST_EQUAL(thrust::log10(a), std::log10(b));
    static_assert(cuda::std::is_same<thrust::complex<T>, decltype(thrust::exp(a))>::value, "");
    static_assert(cuda::std::is_same<thrust::complex<T>, decltype(thrust::log(a))>::value, "");
    static_assert(cuda::std::is_same<thrust::complex<T>, decltype(thrust::log10(a))>::value, "");
  }
};
SimpleUnitTest<TestComplexExponentialFunctions, FloatingPointTypes>
  TestComplexExponentialFunctionsInstance;

template <typename T>
struct TestComplexPowerFunctions
{
  void operator()(void)
  {
    {
      thrust::host_vector<T> data = unittest::random_samples<T>(4);

      const thrust::complex<T> a_thrust(data[0], data[1]);
      const thrust::complex<T> b_thrust(data[2], data[3]);
      const std::complex<T> a_std(a_thrust);
      const std::complex<T> b_std(b_thrust);

      ASSERT_ALMOST_EQUAL(thrust::pow(a_thrust, b_thrust), std::pow(a_std, b_std));
      static_assert(cuda::std::is_same<thrust::complex<T>, decltype(thrust::pow(a_thrust, b_thrust))>::value, "");
      ASSERT_ALMOST_EQUAL(thrust::pow(a_thrust, b_thrust.real()), std::pow(a_std, b_std.real()));
      static_assert(cuda::std::is_same<thrust::complex<T>, decltype(thrust::pow(a_thrust, b_thrust.real()))>::value, "");
      ASSERT_ALMOST_EQUAL(thrust::pow(a_thrust.real(), b_thrust), std::pow(a_std.real(), b_std));
      static_assert(cuda::std::is_same<thrust::complex<T>, decltype(thrust::pow(a_thrust.real(), b_thrust))>::value, "");

      ASSERT_ALMOST_EQUAL(thrust::pow(a_thrust, 4), std::pow(a_std, 4));
      static_assert(cuda::std::is_same<thrust::complex<T>, decltype(thrust::pow(a_thrust, 4))>::value, "");

      ASSERT_ALMOST_EQUAL(thrust::sqrt(a_thrust), std::sqrt(a_std));
      static_assert(cuda::std::is_same<thrust::complex<T>, decltype(thrust::sqrt(a_thrust))>::value, "");
    }

    // Test power functions with promoted types.
    {
      using T0 = T;
      using T1 = other_floating_point_type_t<T0>;
      using promoted = typename thrust::detail::promoted_numerical_type<T0, T1>::type;

      thrust::host_vector<T0> data = unittest::random_samples<T0>(4);

      const thrust::complex<T0> a_thrust(data[0], data[1]);
      const thrust::complex<T1> b_thrust(data[2], data[3]);
      const std::complex<T0> a_std(data[0], data[1]);
      const std::complex<T0> b_std(data[2], data[3]);

      ASSERT_ALMOST_EQUAL(thrust::pow(a_thrust, b_thrust), std::pow(a_std, b_std));
      static_assert(cuda::std::is_same<thrust::complex<promoted>, decltype(thrust::pow(a_thrust, b_thrust))>::value, "");
      ASSERT_ALMOST_EQUAL(thrust::pow(b_thrust, a_thrust), std::pow(b_std, a_std));
      static_assert(cuda::std::is_same<thrust::complex<promoted>, decltype(thrust::pow(b_thrust, a_thrust))>::value, "");
      ASSERT_ALMOST_EQUAL(thrust::pow(a_thrust, b_thrust.real()), std::pow(a_std, b_std.real()));
      static_assert(cuda::std::is_same<thrust::complex<promoted>, decltype(thrust::pow(a_thrust, b_thrust.real()))>::value, "");
      ASSERT_ALMOST_EQUAL(thrust::pow(b_thrust, a_thrust.real()), std::pow(b_std, a_std.real()));
      static_assert(cuda::std::is_same<thrust::complex<promoted>, decltype(thrust::pow(b_thrust, a_thrust.real()))>::value, "");
      ASSERT_ALMOST_EQUAL(thrust::pow(a_thrust.real(), b_thrust), std::pow(a_std.real(), b_std));
      static_assert(cuda::std::is_same<thrust::complex<promoted>, decltype(thrust::pow(a_thrust.real(), b_thrust))>::value, "");
      ASSERT_ALMOST_EQUAL(thrust::pow(b_thrust.real(), a_thrust), std::pow(b_std.real(), a_std));
      static_assert(cuda::std::is_same<thrust::complex<promoted>, decltype(thrust::pow(b_thrust.real(), a_thrust))>::value, "");
    }
  }
};
SimpleUnitTest<TestComplexPowerFunctions, FloatingPointTypes> TestComplexPowerFunctionsInstance;

template <typename T>
struct TestComplexTrigonometricFunctions
{
  void operator()(void)
  {
    thrust::host_vector<T> data = unittest::random_samples<T>(2);

    const thrust::complex<T> a(data[0], data[1]);
    const std::complex<T> c(a);

    ASSERT_ALMOST_EQUAL(thrust::cos(a), std::cos(c));
    ASSERT_ALMOST_EQUAL(thrust::sin(a), std::sin(c));
    ASSERT_ALMOST_EQUAL(thrust::tan(a), std::tan(c));
    static_assert(cuda::std::is_same<thrust::complex<T>, decltype(thrust::cos(a))>::value, "");
    static_assert(cuda::std::is_same<thrust::complex<T>, decltype(thrust::sin(a))>::value, "");
    static_assert(cuda::std::is_same<thrust::complex<T>, decltype(thrust::tan(a))>::value, "");

    ASSERT_ALMOST_EQUAL(thrust::cosh(a), std::cosh(c));
    ASSERT_ALMOST_EQUAL(thrust::sinh(a), std::sinh(c));
    ASSERT_ALMOST_EQUAL(thrust::tanh(a), std::tanh(c));
    static_assert(cuda::std::is_same<thrust::complex<T>, decltype(thrust::cosh(a))>::value, "");
    static_assert(cuda::std::is_same<thrust::complex<T>, decltype(thrust::sinh(a))>::value, "");
    static_assert(cuda::std::is_same<thrust::complex<T>, decltype(thrust::tanh(a))>::value, "");

#if _CCCL_STD_VER >= 2011

    ASSERT_ALMOST_EQUAL(thrust::acos(a), std::acos(c));
    ASSERT_ALMOST_EQUAL(thrust::asin(a), std::asin(c));
    ASSERT_ALMOST_EQUAL(thrust::atan(a), std::atan(c));
    static_assert(cuda::std::is_same<thrust::complex<T>, decltype(thrust::acos(a))>::value, "");
    static_assert(cuda::std::is_same<thrust::complex<T>, decltype(thrust::asin(a))>::value, "");
    static_assert(cuda::std::is_same<thrust::complex<T>, decltype(thrust::atan(a))>::value, "");

    ASSERT_ALMOST_EQUAL(thrust::acosh(a), std::acosh(c));
    ASSERT_ALMOST_EQUAL(thrust::asinh(a), std::asinh(c));
    ASSERT_ALMOST_EQUAL(thrust::atanh(a), std::atanh(c));
    static_assert(cuda::std::is_same<thrust::complex<T>, decltype(thrust::acosh(a))>::value, "");
    static_assert(cuda::std::is_same<thrust::complex<T>, decltype(thrust::asinh(a))>::value, "");
    static_assert(cuda::std::is_same<thrust::complex<T>, decltype(thrust::atanh(a))>::value, "");

#endif
  }
};
SimpleUnitTest<TestComplexTrigonometricFunctions, FloatingPointTypes>
  TestComplexTrigonometricFunctionsInstance;

template <typename T>
struct TestComplexStreamOperators
{
  void operator()(void)
  {
    thrust::host_vector<T> data = unittest::random_samples<T>(2);
    const thrust::complex<T> a(data[0], data[1]);

    std::stringstream out;
    out << a;
    thrust::complex<T> b;
    out >> b;
    ASSERT_ALMOST_EQUAL(a, b);
  }
};
SimpleUnitTest<TestComplexStreamOperators, FloatingPointTypes> TestComplexStreamOperatorsInstance;

#if _CCCL_STD_VER >= 2011
template <typename T>
struct TestComplexStdComplexDeviceInterop
{
  void operator()()
  {
    thrust::host_vector<T> data = unittest::random_samples<T>(6);
    std::vector<std::complex<T>> vec(10);
    vec[0] = std::complex<T>(data[0], data[1]);
    vec[1] = std::complex<T>(data[2], data[3]);
    vec[2] = std::complex<T>(data[4], data[5]);

    thrust::device_vector<thrust::complex<T>> device_vec = vec;
    ASSERT_ALMOST_EQUAL(vec[0].real(), thrust::complex<T>(device_vec[0]).real());
    ASSERT_ALMOST_EQUAL(vec[0].imag(), thrust::complex<T>(device_vec[0]).imag());
    ASSERT_ALMOST_EQUAL(vec[1].real(), thrust::complex<T>(device_vec[1]).real());
    ASSERT_ALMOST_EQUAL(vec[1].imag(), thrust::complex<T>(device_vec[1]).imag());
    ASSERT_ALMOST_EQUAL(vec[2].real(), thrust::complex<T>(device_vec[2]).real());
    ASSERT_ALMOST_EQUAL(vec[2].imag(), thrust::complex<T>(device_vec[2]).imag());
  }
};
SimpleUnitTest<TestComplexStdComplexDeviceInterop, FloatingPointTypes>
  TestComplexStdComplexDeviceInteropInstance;
#endif

template <typename T>
struct TestComplexExplicitConstruction
{
  struct user_complex {
    __host__ __device__ user_complex(T, T) {}
    __host__ __device__ user_complex(const thrust::complex<T>&) {}
  };

  void operator()()
  {
    const thrust::complex<T> input(42.0, 1337.0);
    const user_complex result = thrust::exp(input);
    (void)result;
  }
};
SimpleUnitTest<TestComplexExplicitConstruction, FloatingPointTypes>
  TestComplexExplicitConstructionInstance;
