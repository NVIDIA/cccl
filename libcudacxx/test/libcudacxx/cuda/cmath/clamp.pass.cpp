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

template <class T, class U, class V, class = void>
constexpr bool can_clamp = false;

template <class T, class U, class V>
constexpr bool can_clamp<
  T,
  U,
  V,
  cuda::std::void_t<decltype(cuda::clamp(cuda::std::declval<T>(), cuda::std::declval<U>(), cuda::std::declval<V>()))>> =
  true;

struct NotArithmetic
{
  int value;
};

TEST_FUNC constexpr bool operator<(NotArithmetic lhs, NotArithmetic rhs)
{
  return lhs.value < rhs.value;
}

static_assert(can_clamp<int, int, int>);
static_assert(can_clamp<float, float, float>);
static_assert(!can_clamp<int, int, long>);
static_assert(!can_clamp<double, float, float>);
static_assert(!can_clamp<int*, int*, int*>);
static_assert(!can_clamp<NotArithmetic, NotArithmetic, NotArithmetic>);

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
  static_assert(cuda::std::is_same_v<T, decltype(cuda::clamp(T{}, T{}, T{}))>);
  static_assert(noexcept(cuda::clamp(T{}, T{}, T{})));

  const T one   = make_value<T>(val);
  const T two   = make_value<T>(val + val);
  const T three = make_value<T>(val + val + val);

  assert(is_equal(cuda::clamp(two, one, three), two));
  assert(is_equal(cuda::clamp(one, two, three), two));
  assert(is_equal(cuda::clamp(three, one, two), two));
  assert(is_equal(cuda::clamp(one, one, three), one));
  assert(is_equal(cuda::clamp(three, one, three), three));
  assert(is_equal(cuda::clamp(one, two, two), two));
  assert(is_equal(cuda::clamp(three, two, two), two));

  if constexpr (cuda::std::numeric_limits<T>::is_signed)
  {
    const T minus_one = make_value<T>(-val);
    const T minus_two = make_value<T>(-val - val);
    assert(is_equal(cuda::clamp(minus_two, minus_one, one), minus_one));
    assert(is_equal(cuda::clamp(minus_one, minus_two, one), minus_one));
    assert(is_equal(cuda::clamp(two, minus_two, minus_one), minus_one));
  }

  const T lowest  = cuda::std::numeric_limits<T>::lowest();
  const T highest = cuda::std::numeric_limits<T>::max();
  assert(is_equal(cuda::clamp(lowest, one, two), one));
  assert(is_equal(cuda::clamp(highest, one, two), two));
  assert(is_equal(cuda::clamp(two, lowest, highest), two));

  if constexpr (cuda::std::numeric_limits<T>::has_infinity)
  {
    const T inf       = cuda::std::numeric_limits<T>::infinity();
    const T minus_inf = make_value<T>(-cuda::std::numeric_limits<float>::infinity());
    assert(is_equal(cuda::clamp(inf, one, two), two));
    assert(is_equal(cuda::clamp(minus_inf, one, two), one));
    assert(is_equal(cuda::clamp(two, minus_inf, inf), two));
  }

  // A NaN boundary is ignored and a NaN value is clamped to hi, or to lo if hi is NaN
  if constexpr (cuda::std::numeric_limits<T>::has_quiet_NaN)
  {
    const T nan = cuda::std::numeric_limits<T>::quiet_NaN();
    assert(is_equal(cuda::clamp(nan, one, two), two));
    assert(is_equal(cuda::clamp(nan, one, nan), one));
    assert(is_equal(cuda::clamp(one, nan, two), one));
    assert(is_equal(cuda::clamp(three, nan, two), two));
    assert(is_equal(cuda::clamp(one, two, nan), two));
    assert(is_equal(cuda::clamp(three, two, nan), three));
    assert(is_equal(cuda::clamp(two, nan, nan), two));
  }
}

TEST_FUNC constexpr void test_bool()
{
  static_assert(cuda::std::is_same_v<bool, decltype(cuda::clamp(true, false, true))>);

  assert(cuda::clamp(false, false, true) == false);
  assert(cuda::clamp(true, false, true) == true);
  assert(cuda::clamp(false, true, true) == true);
  assert(cuda::clamp(true, false, false) == false);
}

TEST_FUNC constexpr bool test(float val)
{
  test_bool();

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
#if TEST_CUDA_COMPILER(CLANG)
  // clang drops volatile from local variables, so ptxas sees the value and miscompiles the comparisons of sign extended
  // 8-bit integers. A volatile load through the generic address hides the value from ptxas.
  NV_IF_TARGET(NV_IS_DEVICE, (asm volatile("ld.volatile.f32 %0, [%1];" : "=f"(val) : "l"(&val) : "memory");))
#endif // TEST_CUDA_COMPILER(CLANG)
  test(val);
  test_extended_floating_point(val);
  static_assert(test(1.0f));

  return 0;
}
