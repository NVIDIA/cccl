//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// ADDITIONAL_COMPILE_OPTIONS_HOST: -fext-numeric-literals
// ADDITIONAL_COMPILE_DEFINITIONS: CCCL_GCC_HAS_EXTENDED_NUMERIC_LITERALS

// <cuda/cmath>

// clang-format off
#include <disable_nvfp_conversions_and_operators.h>
// clang-format on

#include <cuda/cmath>
#include <cuda/std/cassert>
#include <cuda/std/cmath>
#include <cuda/std/limits>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

template <class T, class U, class = void>
constexpr bool can_min_max = false;

template <class T, class U>
constexpr bool can_min_max<T,
                           U,
                           cuda::std::void_t<decltype(cuda::min(cuda::std::declval<T>(), cuda::std::declval<U>())),
                                             decltype(cuda::max(cuda::std::declval<T>(), cuda::std::declval<U>()))>> =
  true;

struct NotArithmetic
{
  int value;
};

TEST_FUNC constexpr bool operator<(NotArithmetic lhs, NotArithmetic rhs)
{
  return lhs.value < rhs.value;
}

enum UnscopedEnum
{
};

static_assert(can_min_max<int, int>);
static_assert(can_min_max<float, float>);
static_assert(!can_min_max<int, long>);
static_assert(!can_min_max<short, int>);
static_assert(!can_min_max<int, unsigned>);
static_assert(!can_min_max<unsigned, int>);
static_assert(!can_min_max<float, double>);
static_assert(!can_min_max<double, float>);
static_assert(!can_min_max<int*, int*>);
static_assert(!can_min_max<NotArithmetic, NotArithmetic>);
static_assert(!can_min_max<UnscopedEnum, UnscopedEnum>);

template <class T>
TEST_FUNC constexpr T make_value(float value)
{
  if constexpr (cuda::std::is_integral_v<T>)
  {
    return static_cast<T>(static_cast<int>(value));
  }
  else
  {
    return cuda::std::__fp_cast<T>(value);
  }
}

template <class T>
TEST_FUNC constexpr bool is_equal(T lhs, T rhs)
{
  if constexpr (cuda::std::is_integral_v<T> || cuda::std::is_floating_point_v<T>)
  {
    return lhs == rhs;
  }
  else
  {
    return cuda::std::__fp_get_storage(lhs) == cuda::std::__fp_get_storage(rhs);
  }
}

template <class T>
TEST_FUNC constexpr void test_type(float val)
{
  static_assert(cuda::std::is_same_v<T, decltype(cuda::min(T{}, T{}))>);
  static_assert(cuda::std::is_same_v<T, decltype(cuda::max(T{}, T{}))>);
  static_assert(noexcept(cuda::min(T{}, T{})));
  static_assert(noexcept(cuda::max(T{}, T{})));

  const T one = make_value<T>(val);
  const T two = make_value<T>(val + val);

  assert(is_equal(cuda::min(one, two), one));
  assert(is_equal(cuda::min(two, one), one));
  assert(is_equal(cuda::min(two, two), two));
  assert(is_equal(cuda::max(one, two), two));
  assert(is_equal(cuda::max(two, one), two));
  assert(is_equal(cuda::max(one, one), one));

  if constexpr (cuda::std::numeric_limits<T>::is_signed)
  {
    const T minus_one = make_value<T>(-val);
    assert(is_equal(cuda::min(minus_one, one), minus_one));
    assert(is_equal(cuda::min(one, minus_one), minus_one));
    assert(is_equal(cuda::max(minus_one, one), one));
    assert(is_equal(cuda::max(one, minus_one), one));
  }

  const T lowest  = cuda::std::numeric_limits<T>::lowest();
  const T highest = cuda::std::numeric_limits<T>::max();
  assert(is_equal(cuda::min(lowest, highest), lowest));
  assert(is_equal(cuda::min(highest, lowest), lowest));
  assert(is_equal(cuda::max(lowest, highest), highest));
  assert(is_equal(cuda::max(highest, lowest), highest));

  if constexpr (cuda::std::numeric_limits<T>::has_infinity)
  {
    const T inf       = cuda::std::numeric_limits<T>::infinity();
    const T minus_inf = make_value<T>(-cuda::std::numeric_limits<float>::infinity());
    assert(is_equal(cuda::min(inf, one), one));
    assert(is_equal(cuda::max(inf, one), inf));
    assert(is_equal(cuda::min(minus_inf, one), minus_inf));
    assert(is_equal(cuda::max(minus_inf, one), one));
  }

  // NaN is only returned if both values are NaN
  if constexpr (cuda::std::numeric_limits<T>::has_quiet_NaN)
  {
    const T nan = cuda::std::numeric_limits<T>::quiet_NaN();
    assert(is_equal(cuda::min(nan, one), one));
    assert(is_equal(cuda::min(one, nan), one));
    assert(is_equal(cuda::max(nan, one), one));
    assert(is_equal(cuda::max(one, nan), one));
    assert(cuda::std::isnan(cuda::min(nan, nan)));
    assert(cuda::std::isnan(cuda::max(nan, nan)));
  }
}

TEST_FUNC constexpr void test_bool()
{
  static_assert(cuda::std::is_same_v<bool, decltype(cuda::min(true, false))>);
  static_assert(cuda::std::is_same_v<bool, decltype(cuda::max(true, false))>);

  assert(cuda::min(false, true) == false);
  assert(cuda::min(true, false) == false);
  assert(cuda::min(true, true) == true);
  assert(cuda::max(false, true) == true);
  assert(cuda::max(true, false) == true);
  assert(cuda::max(false, false) == false);
}

TEST_FUNC constexpr void test_explicit_template_argument()
{
  static_assert(cuda::std::is_same_v<long, decltype(cuda::min<long>(1, 2))>);
  static_assert(cuda::std::is_same_v<double, decltype(cuda::max<double>(1.0f, 2))>);

  assert(cuda::min<long>(1, 2) == 1);
  assert(cuda::max<long>(1, 2) == 2);
  assert(cuda::min<double>(1.0f, 2) == 1.0);
  assert(cuda::max<double>(1.0f, 2) == 2.0);
}

TEST_FUNC constexpr bool test(float val)
{
  test_bool();
  test_explicit_template_argument();

  test_type<char>(val);
  test_type<signed char>(val);
  test_type<unsigned char>(val);
  test_type<wchar_t>(val);
#if _CCCL_HAS_CHAR8_T()
  test_type<char8_t>(val);
#endif // _CCCL_HAS_CHAR8_T()
  test_type<char16_t>(val);
  test_type<char32_t>(val);
  test_type<signed short>(val);
  test_type<unsigned short>(val);
  test_type<signed int>(val);
  test_type<unsigned int>(val);
  test_type<signed long>(val);
  test_type<unsigned long>(val);
  test_type<signed long long>(val);
  test_type<unsigned long long>(val);
#if _CCCL_HAS_INT128()
  test_type<__int128_t>(val);
  test_type<__uint128_t>(val);
#endif // _CCCL_HAS_INT128()

  test_type<float>(val);
  test_type<double>(val);
#if _CCCL_HAS_LONG_DOUBLE()
  test_type<long double>(val);
#endif // _CCCL_HAS_LONG_DOUBLE()

  return true;
}

TEST_FUNC void test_extended_floating_point(float val)
{
#if _CCCL_HAS_NVFP16()
  test_type<__half>(val);
#endif // _CCCL_HAS_NVFP16()
#if _CCCL_HAS_NVBF16()
  test_type<__nv_bfloat16>(val);
#endif // _CCCL_HAS_NVBF16()
#if _CCCL_HAS_NVFP8_E4M3()
  test_type<__nv_fp8_e4m3>(val);
#endif // _CCCL_HAS_NVFP8_E4M3()
#if _CCCL_HAS_NVFP8_E5M2()
  test_type<__nv_fp8_e5m2>(val);
#endif // _CCCL_HAS_NVFP8_E5M2()
#if _CCCL_HAS_NVFP8_UE5M3()
  test_type<__nv_fp8_ue5m3>(val);
#endif // _CCCL_HAS_NVFP8_UE5M3()
#if _CCCL_HAS_NVFP8_E8M0()
  test_type<__nv_fp8_e8m0>(val);
#endif // _CCCL_HAS_NVFP8_E8M0()
#if _CCCL_HAS_NVFP6_E2M3()
  test_type<__nv_fp6_e2m3>(val);
#endif // _CCCL_HAS_NVFP6_E2M3()
#if _CCCL_HAS_NVFP6_E3M2()
  test_type<__nv_fp6_e3m2>(val);
#endif // _CCCL_HAS_NVFP6_E3M2()
#if _CCCL_HAS_NVFP4_E2M1()
  test_type<__nv_fp4_e2m1>(val);
#endif // _CCCL_HAS_NVFP4_E2M1()
#if _CCCL_HAS_FLOAT128()
  test_type<__float128>(val);
#endif // _CCCL_HAS_FLOAT128()
}

int main(int, char**)
{
  volatile float val = 1.0f;
  test(val);
  test_extended_floating_point(val);
  static_assert(test(1.0f));

  return 0;
}
