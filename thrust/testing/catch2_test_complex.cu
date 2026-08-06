#include <thrust/complex.h>
#include <thrust/host_vector.h>

#include <complex>
#include <iostream>
#include <sstream>
#include <type_traits>
#include <vector>

#include "catch2_test_helper.h"
#include <unittest/random.h>
#include <unittest/testframework.h>

_CCCL_DIAG_SUPPRESS_MSVC(4244) // conversion from 'const T1' to 'const T', possible loss of data

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

// Helper to compare complex numbers with approximate equality
// Supports both scalar and thrust::complex<T> types
double const DEFAULT_RELATIVE_TOL = 1e-4;
double const DEFAULT_ABSOLUTE_TOL = 1e-4;

template <typename T>
inline constexpr bool is_complex = false;
template <typename T>
inline constexpr bool is_complex<thrust::complex<T>> = true;
template <typename T>
inline constexpr bool is_complex<std::complex<T>> = true;

// Overload for complex types
template <typename T1, typename T2>
::cuda::std::enable_if_t<is_complex<T1> && is_complex<T2>> require_almost_equal(const T1& a, const T2& b)
{
  CHECK(a.real() == Catch::Approx(b.real()).margin(DEFAULT_ABSOLUTE_TOL).epsilon(DEFAULT_RELATIVE_TOL));
  CHECK(a.imag() == Catch::Approx(b.imag()).margin(DEFAULT_ABSOLUTE_TOL).epsilon(DEFAULT_RELATIVE_TOL));
}

template <typename T1, typename T2>
::cuda::std::enable_if_t<!is_complex<T1> && !is_complex<T2>> require_almost_equal(const T1& a, const T2& b)
{
  CHECK(a == Catch::Approx(b).margin(DEFAULT_ABSOLUTE_TOL).epsilon(DEFAULT_RELATIVE_TOL));
}
} // anonymous namespace

TEMPLATE_LIST_TEST_CASE("ComplexSizeAndAlignment", "[complex]", FloatingPointTypes)
{
  using T = TestType;

  STATIC_REQUIRE(sizeof(thrust::complex<T>) == sizeof(T) * 2);
  STATIC_REQUIRE(alignof(thrust::complex<T>) == alignof(T) * 2);

  STATIC_REQUIRE(sizeof(thrust::complex<T const>) == sizeof(T) * 2);
  STATIC_REQUIRE(alignof(thrust::complex<T const>) == alignof(T) * 2);
}

TEMPLATE_LIST_TEST_CASE("ComplexConstructionAndAssignment", "[complex]", FloatingPointTypes)
{
  using T = TestType;

  thrust::host_vector<T> data = unittest::random_samples<T>(2);

  const T real = data[0];
  const T imag = data[1];

  {
    const thrust::complex<T> construct_from_real_and_imag(real, imag);
    CHECK(real == construct_from_real_and_imag.real());
    CHECK(imag == construct_from_real_and_imag.imag());
  }

  {
    const thrust::complex<T> construct_from_real(real);
    CHECK(real == construct_from_real.real());
    CHECK(T() == construct_from_real.imag());
  }

  {
    const thrust::complex<T> expected(real, imag);
    thrust::complex<T> construct_from_copy(expected);
    CHECK(expected.real() == construct_from_copy.real());
    CHECK(expected.imag() == construct_from_copy.imag());
  }

  {
    thrust::complex<T> construct_from_move(thrust::complex<T>(real, imag));
    CHECK(real == construct_from_move.real());
    CHECK(imag == construct_from_move.imag());
  }

  {
    thrust::complex<T> copy_assign{};
    const thrust::complex<T> expected{real, imag};
    copy_assign = expected;
    CHECK(expected.real() == copy_assign.real());
    CHECK(expected.imag() == copy_assign.imag());
  }

  {
    thrust::complex<T> move_assign{};
    const thrust::complex<T> expected{real, imag};
    move_assign = thrust::complex<T>{real, imag};
    CHECK(expected.real() == move_assign.real());
    CHECK(expected.imag() == move_assign.imag());
  }

  {
    thrust::complex<T> assign_from_lvalue_T{};
    const thrust::complex<T> expected{real};
    const T to_be_copied = real;
    assign_from_lvalue_T = to_be_copied;
    CHECK(expected.real() == assign_from_lvalue_T.real());
    CHECK(expected.imag() == assign_from_lvalue_T.imag());
  }

  {
    thrust::complex<T> assign_from_rvalue_T{};
    const thrust::complex<T> expected{real};
    assign_from_rvalue_T = T(real);
    CHECK(expected.real() == assign_from_rvalue_T.real());
    CHECK(expected.imag() == assign_from_rvalue_T.imag());
  }

  {
    const std::complex<T> expected(real, imag);
    const thrust::complex<T> copy_from_std(expected);
    CHECK(expected.real() == copy_from_std.real());
    CHECK(expected.imag() == copy_from_std.imag());
  }

  {
    thrust::complex<T> assign_from_lvalue_std{};
    const std::complex<T> expected(real, imag);
    assign_from_lvalue_std = expected;
    CHECK(expected.real() == assign_from_lvalue_std.real());
    CHECK(expected.imag() == assign_from_lvalue_std.imag());
  }

  {
    thrust::complex<T> assign_from_rvalue_std{};
    assign_from_rvalue_std = std::complex<T>(real, imag);
    CHECK(real == assign_from_rvalue_std.real());
    CHECK(imag == assign_from_rvalue_std.imag());
  }
}

TEMPLATE_LIST_TEST_CASE("ComplexConstructionAndAssignmentWithPromoting", "[complex]", FloatingPointTypes)
{
  using T  = TestType;
  using T0 = T;
  using T1 = other_floating_point_type_t<T0>;

  thrust::host_vector<T0> data = unittest::random_samples<T0>(2);

  const T0 real_T0 = data[0];
  const T0 imag_T0 = data[1];

  const T1 real_T1 = static_cast<T1>(real_T0);
  const T1 imag_T1 = static_cast<T1>(imag_T0);

  {
    const thrust::complex<T0> construct_from_real_and_imag(real_T1, imag_T1);
    require_almost_equal(real_T0, construct_from_real_and_imag.real());
    require_almost_equal(imag_T0, construct_from_real_and_imag.imag());
  }

  {
    const thrust::complex<T0> construct_from_real(real_T1);
    require_almost_equal(real_T0, construct_from_real.real());
    CHECK(T0() == construct_from_real.imag());
  }

  {
    const thrust::complex<T1> expected(real_T1, imag_T1);
    thrust::complex<T0> construct_from_copy(expected);
    require_almost_equal(real_T0, construct_from_copy.real());
    require_almost_equal(imag_T0, construct_from_copy.imag());
  }

  {
    thrust::complex<T0> construct_from_move(thrust::complex<T1>(real_T1, imag_T1));
    require_almost_equal(real_T0, construct_from_move.real());
    require_almost_equal(imag_T0, construct_from_move.imag());
  }

  {
    thrust::complex<T0> copy_assign{};
    const thrust::complex<T1> expected{real_T1, imag_T1};
    copy_assign = expected;
    CHECK(expected.real() == copy_assign.real());
    CHECK(expected.imag() == copy_assign.imag());
  }

  {
    thrust::complex<T0> assign_from_lvalue_T{};
    const thrust::complex<T1> expected{real_T1};
    const T1 to_be_copied = real_T1;
    assign_from_lvalue_T  = to_be_copied;
    require_almost_equal(expected.real(), assign_from_lvalue_T.real());
    CHECK(expected.imag() == assign_from_lvalue_T.imag());
  }

  {
    const std::complex<T1> expected(real_T1, imag_T1);
    const thrust::complex<T0> copy_from_std(expected);
    require_almost_equal(expected.real(), copy_from_std.real());
    require_almost_equal(expected.imag(), copy_from_std.imag());
  }

  {
    thrust::complex<T1> assign_from_lvalue_std{};
    const std::complex<T0> expected(real_T1, imag_T1);
    assign_from_lvalue_std = expected;
    require_almost_equal(expected.real(), assign_from_lvalue_std.real());
    require_almost_equal(expected.imag(), assign_from_lvalue_std.imag());
  }

  {
    thrust::complex<T0> assign_from_rvalue_std{};
    assign_from_rvalue_std = std::complex<T1>(real_T1, imag_T1);
    require_almost_equal(real_T0, assign_from_rvalue_std.real());
    require_almost_equal(imag_T1, assign_from_rvalue_std.imag());
  }
}

TEMPLATE_LIST_TEST_CASE("ComplexGetters", "[complex]", FloatingPointTypes)
{
  using T = TestType;

  thrust::host_vector<T> data = unittest::random_samples<T>(2);

  thrust::complex<T> z(data[0], data[1]);

  CHECK(data[0] == z.real());
  CHECK(data[1] == z.imag());

  z.real(data[1]);
  z.imag(data[0]);
  CHECK(data[1] == z.real());
  CHECK(data[0] == z.imag());

  volatile thrust::complex<T> v(data[0], data[1]);

  CHECK(data[0] == v.real());
  CHECK(data[1] == v.imag());

  v.real(data[1]);
  v.imag(data[0]);
  CHECK(data[1] == v.real());
  CHECK(data[0] == v.imag());
}

TEMPLATE_LIST_TEST_CASE("ComplexComparisionOperators", "[complex]", FloatingPointTypes)
{
  using T = TestType;

  {
    thrust::host_vector<T> data = unittest::random_samples<T>(1);

    const T a = data[0];
    const T b = data[0] + T(1.0);

    CHECK(thrust::complex<T>(a, b) == thrust::complex<T>(a, b));
    CHECK(thrust::complex<T>(a, T()) == a);
    CHECK(a == thrust::complex<T>(a, T()));

    CHECK(thrust::complex<T>(a, b) == std::complex<T>(a, b));
    CHECK(std::complex<T>(a, b) == thrust::complex<T>(a, b));

    CHECK(thrust::complex<T>(a, b) != thrust::complex<T>(b, a));
    CHECK(thrust::complex<T>(a, T()) != b);
    CHECK(b != thrust::complex<T>(a, T()));

    CHECK(thrust::complex<T>(a, b) != a);
    CHECK(a != thrust::complex<T>(a, b));

    CHECK(thrust::complex<T>(a, b) != std::complex<T>(b, a));
    CHECK(std::complex<T>(a, b) != thrust::complex<T>(b, a));
  }

  // Testing comparison operators with promoted types.
  // These tests don't use random numbers on purpose since T0(x) == T1(x) will
  // not be true for all x.
  {
    using T0 = T;
    using T1 = other_floating_point_type_t<T0>;

    CHECK(thrust::complex<T0>(1.0, 2.0) == thrust::complex<T1>(1.0, 2.0));
    CHECK(thrust::complex<T0>(1.0, T0()) == T1(1.0));
    CHECK(T1(1.0) == thrust::complex<T0>(1.0, 0.0));

    CHECK(thrust::complex<T0>(1.0, 2.0) == std::complex<T1>(1.0, 2.0));
    CHECK(std::complex<T0>(1.0, 2.0) == thrust::complex<T1>(1.0, 2.0));

    CHECK(thrust::complex<T0>(1.0, 2.0) != T1(1.0));
    CHECK(T1(1.0) != thrust::complex<T0>(1.0, 2.0));
  }
}

TEMPLATE_LIST_TEST_CASE("ComplexMemberOperators", "[complex]", FloatingPointTypes)
{
  using T = TestType;

  {
    thrust::host_vector<T> data = unittest::random_samples<T>(5);

    thrust::complex<T> a_thrust(data[0], data[1]);
    const thrust::complex<T> b_thrust(data[2], data[3]);

    std::complex<T> a_std(a_thrust);
    const std::complex<T> b_std(b_thrust);

    a_thrust += b_thrust;
    a_std += b_std;
    require_almost_equal(a_thrust, a_std);

    a_thrust -= b_thrust;
    a_std -= b_std;
    require_almost_equal(a_thrust, a_std);

    a_thrust *= b_thrust;
    a_std *= b_std;
    require_almost_equal(a_thrust, a_std);

    a_thrust /= b_thrust;
    a_std /= b_std;
    require_almost_equal(a_thrust, a_std);

    // arithmetic operators with `double` and `float`
    const T real = data[4];

    a_thrust += real;
    a_std += std::complex<T>(real);
    require_almost_equal(a_thrust, a_std);

    a_thrust -= real;
    a_std -= std::complex<T>(real);
    require_almost_equal(a_thrust, a_std);

    a_thrust *= real;
    a_std *= std::complex<T>(real);
    require_almost_equal(a_thrust, a_std);

    a_thrust /= real;
    a_std /= std::complex<T>(real);
    require_almost_equal(a_thrust, a_std);

    // casting operator
    a_std = (std::complex<T>) a_thrust;
    require_almost_equal(a_thrust.real(), a_std.real());
    require_almost_equal(a_thrust.imag(), a_std.imag());
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
    require_almost_equal(a_thrust, a_std);
    require_almost_equal(b_thrust, b_std);

    a_thrust += b_thrust;
    a_std += b_std;
    require_almost_equal(a_thrust, a_std);

    a_thrust -= b_thrust;
    a_std -= b_std;
    require_almost_equal(a_thrust, a_std);

    a_thrust *= b_thrust;
    a_std *= b_std;
    require_almost_equal(a_thrust, a_std);

    a_thrust /= b_thrust;
    a_std /= b_std;
    require_almost_equal(a_thrust, a_std);

    // Testing arithmetic member operators with another floating point type.
    const T1 e = data[2];

    a_thrust += e;
    a_std += std::complex<T1>(e);
    require_almost_equal(a_thrust, a_std);

    a_thrust -= e;
    a_std -= std::complex<T1>(e);
    require_almost_equal(a_thrust, a_std);

    a_thrust *= e;
    a_std *= std::complex<T1>(e);
    require_almost_equal(a_thrust, a_std);

    a_thrust /= e;
    a_std /= std::complex<T1>(e);
    require_almost_equal(a_thrust, a_std);
  }
}

// Test the basic arithmetic functions against std
TEMPLATE_LIST_TEST_CASE("ComplexBasicArithmetic", "[complex]", FloatingPointTypes)
{
  using T = TestType;

  thrust::host_vector<T> data = unittest::random_samples<T>(2);

  const thrust::complex<T> a(data[0], data[1]);
  const std::complex<T> b(a);

  // Test the basic arithmetic functions against std
  require_almost_equal(thrust::abs(a), std::abs(b));
  require_almost_equal(thrust::arg(a), std::arg(b));
  require_almost_equal(thrust::norm(a), std::norm(b));

  CHECK(thrust::conj(a) == std::conj(b));
  STATIC_REQUIRE(cuda::std::is_same<thrust::complex<T>, decltype(thrust::conj(a))>::value);

  require_almost_equal(thrust::polar(data[0], data[1]), std::polar(data[0], data[1]));
  STATIC_REQUIRE(cuda::std::is_same<thrust::complex<T>, decltype(thrust::polar(data[0], data[1]))>::value);

  // random_samples does not seem to produce infinities so proj(z) == z
  CHECK(thrust::proj(a) == a);
  STATIC_REQUIRE(cuda::std::is_same<thrust::complex<T>, decltype(thrust::proj(a))>::value);
}

TEMPLATE_LIST_TEST_CASE("ComplexBinaryArithmetic", "[complex]", FloatingPointTypes)
{
  using T = TestType;

  {
    thrust::host_vector<T> data = unittest::random_samples<T>(5);

    const thrust::complex<T> a(data[0], data[1]);
    const thrust::complex<T> b(data[2], data[3]);
    const T real = data[4];

    require_almost_equal(a * b, std::complex<T>(a) * std::complex<T>(b));
    require_almost_equal(a * real, std::complex<T>(a) * real);
    require_almost_equal(real * b, real * std::complex<T>(b));

    require_almost_equal(a / b, std::complex<T>(a) / std::complex<T>(b));
    require_almost_equal(a / real, std::complex<T>(a) / real);
    require_almost_equal(real / b, real / std::complex<T>(b));

    require_almost_equal(a + b, std::complex<T>(a) + std::complex<T>(b));
    require_almost_equal(a + real, std::complex<T>(a) + real);
    require_almost_equal(real + b, real + std::complex<T>(b));

    require_almost_equal(a - b, std::complex<T>(a) - std::complex<T>(b));
    require_almost_equal(a - real, std::complex<T>(a) - real);
    require_almost_equal(real - b, real - std::complex<T>(b));
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

    require_almost_equal(a_thrust * b_thrust, a_std * b_std);
    require_almost_equal(a_thrust * real_T1, a_std * real_T0);
    require_almost_equal(real_T0 * b_thrust, real_T0 * b_std);

    require_almost_equal(a_thrust / b_thrust, a_std / b_std);
    require_almost_equal(a_thrust / real_T1, a_std / real_T0);
    require_almost_equal(real_T0 / b_thrust, real_T0 / b_std);

    require_almost_equal(a_thrust + b_thrust, a_std + b_std);
    require_almost_equal(a_thrust + real_T1, a_std + real_T0);
    require_almost_equal(real_T0 + b_thrust, real_T0 + b_std);

    require_almost_equal(a_thrust - b_thrust, a_std - b_std);
    require_almost_equal(a_thrust - real_T1, a_std - real_T0);
    require_almost_equal(real_T0 - b_thrust, real_T0 - b_std);
  }
}

TEMPLATE_LIST_TEST_CASE("ComplexUnaryArithmetic", "[complex]", FloatingPointTypes)
{
  using T = TestType;

  thrust::host_vector<T> data = unittest::random_samples<T>(2);

  const thrust::complex<T> a(data[0], data[1]);

  CHECK(+a == a);
  CHECK(-a == a * (-1.0));
}

TEMPLATE_LIST_TEST_CASE("ComplexExponentialFunctions", "[complex]", FloatingPointTypes)
{
  using T = TestType;

  thrust::host_vector<T> data = unittest::random_samples<T>(2);

  const thrust::complex<T> a(data[0], data[1]);
  const std::complex<T> b(a);

  require_almost_equal(thrust::exp(a), std::exp(b));
  require_almost_equal(thrust::log(a), std::log(b));
  require_almost_equal(thrust::log10(a), std::log10(b));
  STATIC_REQUIRE(cuda::std::is_same<thrust::complex<T>, decltype(thrust::exp(a))>::value);
  STATIC_REQUIRE(cuda::std::is_same<thrust::complex<T>, decltype(thrust::log(a))>::value);
  STATIC_REQUIRE(cuda::std::is_same<thrust::complex<T>, decltype(thrust::log10(a))>::value);
}

TEMPLATE_LIST_TEST_CASE("ComplexPowerFunctions", "[complex]", FloatingPointTypes)
{
  using T = TestType;

  {
    thrust::host_vector<T> data = unittest::random_samples<T>(4);

    const thrust::complex<T> a_thrust(data[0], data[1]);
    const thrust::complex<T> b_thrust(data[2], data[3]);
    const std::complex<T> a_std(a_thrust);
    const std::complex<T> b_std(b_thrust);

    require_almost_equal(thrust::pow(a_thrust, b_thrust), std::pow(a_std, b_std));
    STATIC_REQUIRE(cuda::std::is_same<thrust::complex<T>, decltype(thrust::pow(a_thrust, b_thrust))>::value);
    require_almost_equal(thrust::pow(a_thrust, b_thrust.real()), std::pow(a_std, b_std.real()));
    STATIC_REQUIRE(cuda::std::is_same<thrust::complex<T>, decltype(thrust::pow(a_thrust, b_thrust.real()))>::value);
    require_almost_equal(thrust::pow(a_thrust.real(), b_thrust), std::pow(a_std.real(), b_std));
    STATIC_REQUIRE(cuda::std::is_same<thrust::complex<T>, decltype(thrust::pow(a_thrust.real(), b_thrust))>::value);

    require_almost_equal(thrust::pow(a_thrust, 4), std::pow(a_std, 4));
    STATIC_REQUIRE(cuda::std::is_same<thrust::complex<T>, decltype(thrust::pow(a_thrust, 4))>::value);

    require_almost_equal(thrust::sqrt(a_thrust), std::sqrt(a_std));
    STATIC_REQUIRE(cuda::std::is_same<thrust::complex<T>, decltype(thrust::sqrt(a_thrust))>::value);
  }

  // Test power functions with promoted types.
  {
    using T0       = T;
    using T1       = other_floating_point_type_t<T0>;
    using promoted = ::cuda::std::common_type_t<T0, T1>;

    thrust::host_vector<T0> data = unittest::random_samples<T0>(4);

    const thrust::complex<T0> a_thrust(data[0], data[1]);
    const thrust::complex<T1> b_thrust(data[2], data[3]);
    const std::complex<T0> a_std(data[0], data[1]);
    const std::complex<T0> b_std(data[2], data[3]);

    require_almost_equal(thrust::pow(a_thrust, b_thrust), std::pow(a_std, b_std));
    STATIC_REQUIRE(cuda::std::is_same<thrust::complex<promoted>, decltype(thrust::pow(a_thrust, b_thrust))>::value);
    require_almost_equal(thrust::pow(b_thrust, a_thrust), std::pow(b_std, a_std));
    STATIC_REQUIRE(cuda::std::is_same<thrust::complex<promoted>, decltype(thrust::pow(b_thrust, a_thrust))>::value);
    require_almost_equal(thrust::pow(a_thrust, b_thrust.real()), std::pow(a_std, b_std.real()));
    STATIC_REQUIRE(
      cuda::std::is_same<thrust::complex<promoted>, decltype(thrust::pow(a_thrust, b_thrust.real()))>::value);
    require_almost_equal(thrust::pow(b_thrust, a_thrust.real()), std::pow(b_std, a_std.real()));
    STATIC_REQUIRE(
      cuda::std::is_same<thrust::complex<promoted>, decltype(thrust::pow(b_thrust, a_thrust.real()))>::value);
    require_almost_equal(thrust::pow(a_thrust.real(), b_thrust), std::pow(a_std.real(), b_std));
    STATIC_REQUIRE(
      cuda::std::is_same<thrust::complex<promoted>, decltype(thrust::pow(a_thrust.real(), b_thrust))>::value);
    require_almost_equal(thrust::pow(b_thrust.real(), a_thrust), std::pow(b_std.real(), a_std));
    STATIC_REQUIRE(
      cuda::std::is_same<thrust::complex<promoted>, decltype(thrust::pow(b_thrust.real(), a_thrust))>::value);
  }
}

TEMPLATE_LIST_TEST_CASE("ComplexTrigonometricFunctions", "[complex]", FloatingPointTypes)
{
  using T = TestType;

  thrust::host_vector<T> data = unittest::random_samples<T>(2);

  const thrust::complex<T> a(data[0], data[1]);
  const std::complex<T> c(a);

  require_almost_equal(thrust::cos(a), std::cos(c));
  require_almost_equal(thrust::sin(a), std::sin(c));
  require_almost_equal(thrust::tan(a), std::tan(c));
  STATIC_REQUIRE(cuda::std::is_same<thrust::complex<T>, decltype(thrust::cos(a))>::value);
  STATIC_REQUIRE(cuda::std::is_same<thrust::complex<T>, decltype(thrust::sin(a))>::value);
  STATIC_REQUIRE(cuda::std::is_same<thrust::complex<T>, decltype(thrust::tan(a))>::value);

  require_almost_equal(thrust::cosh(a), std::cosh(c));
  require_almost_equal(thrust::sinh(a), std::sinh(c));
  require_almost_equal(thrust::tanh(a), std::tanh(c));
  STATIC_REQUIRE(cuda::std::is_same<thrust::complex<T>, decltype(thrust::cosh(a))>::value);
  STATIC_REQUIRE(cuda::std::is_same<thrust::complex<T>, decltype(thrust::sinh(a))>::value);
  STATIC_REQUIRE(cuda::std::is_same<thrust::complex<T>, decltype(thrust::tanh(a))>::value);

  require_almost_equal(thrust::acos(a), std::acos(c));
  require_almost_equal(thrust::asin(a), std::asin(c));
  require_almost_equal(thrust::atan(a), std::atan(c));
  STATIC_REQUIRE(cuda::std::is_same<thrust::complex<T>, decltype(thrust::acos(a))>::value);
  STATIC_REQUIRE(cuda::std::is_same<thrust::complex<T>, decltype(thrust::asin(a))>::value);
  STATIC_REQUIRE(cuda::std::is_same<thrust::complex<T>, decltype(thrust::atan(a))>::value);

  require_almost_equal(thrust::acosh(a), std::acosh(c));
  require_almost_equal(thrust::asinh(a), std::asinh(c));
  require_almost_equal(thrust::atanh(a), std::atanh(c));
  STATIC_REQUIRE(cuda::std::is_same<thrust::complex<T>, decltype(thrust::acosh(a))>::value);
  STATIC_REQUIRE(cuda::std::is_same<thrust::complex<T>, decltype(thrust::asinh(a))>::value);
  STATIC_REQUIRE(cuda::std::is_same<thrust::complex<T>, decltype(thrust::atanh(a))>::value);
}

TEMPLATE_LIST_TEST_CASE("ComplexStreamOperators", "[complex]", FloatingPointTypes)
{
  using T = TestType;

  thrust::host_vector<T> data = unittest::random_samples<T>(2);
  const thrust::complex<T> a(data[0], data[1]);

  std::stringstream out;
  out << a;
  thrust::complex<T> b{};
  out >> b;
  require_almost_equal(a, b);
}

TEMPLATE_LIST_TEST_CASE("ComplexStdComplexDeviceInterop", "[complex]", FloatingPointTypes)
{
  using T = TestType;

  thrust::host_vector<T> data = unittest::random_samples<T>(6);
  std::vector<std::complex<T>> vec(10);
  vec[0] = std::complex<T>(data[0], data[1]);
  vec[1] = std::complex<T>(data[2], data[3]);
  vec[2] = std::complex<T>(data[4], data[5]);

  thrust::device_vector<thrust::complex<T>> device_vec = vec;
  require_almost_equal(vec[0].real(), thrust::complex<T>(device_vec[0]).real());
  require_almost_equal(vec[0].imag(), thrust::complex<T>(device_vec[0]).imag());
  require_almost_equal(vec[1].real(), thrust::complex<T>(device_vec[1]).real());
  require_almost_equal(vec[1].imag(), thrust::complex<T>(device_vec[1]).imag());
  require_almost_equal(vec[2].real(), thrust::complex<T>(device_vec[2]).real());
  require_almost_equal(vec[2].imag(), thrust::complex<T>(device_vec[2]).imag());
}

TEMPLATE_LIST_TEST_CASE("ComplexExplicitConstruction", "[complex]", FloatingPointTypes)
{
  using T = TestType;

  struct user_complex
  {
    _CCCL_HOST_DEVICE user_complex(T, T) {}
    _CCCL_HOST_DEVICE user_complex(const thrust::complex<T>&) {}
  };

  const thrust::complex<T> input(42.0, 1337.0);
  [[maybe_unused]] const user_complex result = thrust::exp(input);
}

// Compile-time tests for thrust::complex (issue #485). These exercise the
// constexpr-marked constructors, accessors, arithmetic operators and equality
// operators of thrust::complex<T> via static_assert. They are independent of
// the runtime test cases above and run at compile time only.
namespace
{
template <typename T>
constexpr bool test_complex_constexpr_construction()
{
  using C = thrust::complex<T>;

  // Default construction zero-initializes.
  static_assert(C().real() == T(0), "default ctor real");
  static_assert(C().imag() == T(0), "default ctor imag");

  // Construct from real only.
  static_assert(C(T(2)).real() == T(2), "ctor(real)");
  static_assert(C(T(2)).imag() == T(0), "ctor(real) imag is zero");

  // Construct from real and imag.
  static_assert(C(T(1), T(2)).real() == T(1), "ctor(real, imag) real");
  static_assert(C(T(1), T(2)).imag() == T(2), "ctor(real, imag) imag");

  // Copy and converting copy.
  static_assert(C(C(T(3), T(4))).real() == T(3), "copy ctor real");
  static_assert(C(C(T(3), T(4))).imag() == T(4), "copy ctor imag");
  return true;
}

template <typename T>
constexpr bool test_complex_constexpr_arithmetic()
{
  using C = thrust::complex<T>;

  // Binary +, -, * with two complex operands.
  static_assert((C(T(1), T(2)) + C(T(3), T(4))).real() == T(4), "operator+ real");
  static_assert((C(T(1), T(2)) + C(T(3), T(4))).imag() == T(6), "operator+ imag");
  static_assert((C(T(5), T(6)) - C(T(1), T(2))).real() == T(4), "operator- real");
  static_assert((C(T(5), T(6)) - C(T(1), T(2))).imag() == T(4), "operator- imag");
  // (1+2i)(3+4i) = (1*3 - 2*4) + (1*4 + 2*3)i = -5 + 10i
  static_assert((C(T(1), T(2)) * C(T(3), T(4))).real() == T(-5), "operator* real");
  static_assert((C(T(1), T(2)) * C(T(3), T(4))).imag() == T(10), "operator* imag");

  // Mixed scalar/complex operands.
  static_assert((C(T(1), T(2)) + T(1)).real() == T(2), "operator+ scalar rhs");
  static_assert((T(1) + C(T(1), T(2))).real() == T(2), "operator+ scalar lhs");
  static_assert((C(T(2), T(4)) * T(2)).real() == T(4), "operator* scalar rhs");
  static_assert((T(2) * C(T(2), T(4))).imag() == T(8), "operator* scalar lhs");
  static_assert((C(T(4), T(8)) / T(2)).real() == T(2), "operator/ scalar rhs");
  static_assert((C(T(4), T(8)) / T(2)).imag() == T(4), "operator/ scalar rhs imag");

  // Complex / complex (the hardest one to keep constexpr-friendly).
  static_assert((C(T(1), T(0)) / C(T(1), T(0))).real() == T(1), "complex/complex");

  // Unary + and -.
  static_assert((+C(T(1), T(2))).real() == T(1), "unary plus real");
  static_assert((+C(T(1), T(2))).imag() == T(2), "unary plus imag");
  static_assert((-C(T(1), T(2))).real() == T(-1), "unary minus real");
  static_assert((-C(T(1), T(2))).imag() == T(-2), "unary minus imag");

  // conj is constexpr.
  static_assert(thrust::conj(C(T(1), T(2))).real() == T(1), "conj real");
  static_assert(thrust::conj(C(T(1), T(2))).imag() == T(-2), "conj imag");

  return true;
}

template <typename T>
constexpr bool test_complex_constexpr_equality()
{
  using C = thrust::complex<T>;

  static_assert(C(T(1), T(2)) == C(T(1), T(2)), "operator== same");
  static_assert(C(T(1), T(2)) != C(T(3), T(4)), "operator!= different");
  static_assert(C(T(5), T(0)) == T(5), "operator== complex/scalar");
  static_assert(T(5) == C(T(5), T(0)), "operator== scalar/complex");
  static_assert(C(T(5), T(1)) != T(5), "operator!= complex/scalar (imag != 0)");
  static_assert(T(5) != C(T(6), T(0)), "operator!= scalar/complex");

  return true;
}

template <typename T>
constexpr bool test_complex_constexpr_compound_assignment()
{
  using C = thrust::complex<T>;

  // The compound assignment operators mutate state; exercise them inside
  // a constexpr function so we can verify the final value via static_assert.
  C v(T(1), T(2));
  v += C(T(3), T(4));
  if (v.real() != T(4) || v.imag() != T(6))
  {
    return false;
  }

  v -= C(T(3), T(4));
  if (v.real() != T(1) || v.imag() != T(2))
  {
    return false;
  }

  v *= C(T(3), T(4));
  // (1+2i)(3+4i) = -5 + 10i
  if (v.real() != T(-5) || v.imag() != T(10))
  {
    return false;
  }

  // Scalar compound assignments.
  C s(T(2), T(4));
  s += T(1);
  if (s.real() != T(3) || s.imag() != T(4))
  {
    return false;
  }
  s *= T(2);
  if (s.real() != T(6) || s.imag() != T(8))
  {
    return false;
  }
  s /= T(2);
  if (s.real() != T(3) || s.imag() != T(4))
  {
    return false;
  }

  return true;
}

template <typename T>
constexpr bool test_complex_constexpr_setters()
{
  using C = thrust::complex<T>;
  C v;
  v.real(T(5));
  v.imag(T(7));
  return v.real() == T(5) && v.imag() == T(7);
}

// Force evaluation of the static_asserts inside the helpers above, for both
// float and double, at compile time.
static_assert(test_complex_constexpr_construction<float>(), "");
static_assert(test_complex_constexpr_construction<double>(), "");
static_assert(test_complex_constexpr_arithmetic<float>(), "");
static_assert(test_complex_constexpr_arithmetic<double>(), "");
static_assert(test_complex_constexpr_equality<float>(), "");
static_assert(test_complex_constexpr_equality<double>(), "");
static_assert(test_complex_constexpr_compound_assignment<float>(), "");
static_assert(test_complex_constexpr_compound_assignment<double>(), "");
static_assert(test_complex_constexpr_setters<float>(), "");
static_assert(test_complex_constexpr_setters<double>(), "");
} // anonymous namespace

TEMPLATE_LIST_TEST_CASE("ComplexConstexpr", "[complex]", FloatingPointTypes)
{
  // The actual checks are static_asserts at namespace scope above; this
  // runtime case exists so the test binary records that the constexpr
  // checks were exercised for each floating point type.
  using T = TestType;
  STATIC_REQUIRE(test_complex_constexpr_construction<T>());
  STATIC_REQUIRE(test_complex_constexpr_arithmetic<T>());
  STATIC_REQUIRE(test_complex_constexpr_equality<T>());
  STATIC_REQUIRE(test_complex_constexpr_compound_assignment<T>());
  STATIC_REQUIRE(test_complex_constexpr_setters<T>());
}
