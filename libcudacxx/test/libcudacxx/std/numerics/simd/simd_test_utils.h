//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef SIMD_TEST_UTILS_H
#define SIMD_TEST_UTILS_H

#include <cuda/std/__simd_>
#include <cuda/std/array>
#include <cuda/std/cmath>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

#include "test_macros.h"

namespace simd = cuda::std::simd;

//----------------------------------------------------------------------------------------------------------------------
// common utilities

struct wrong_generator
{};

template <typename>
struct is_const_member_function : cuda::std::false_type
{};

template <typename R, typename C, typename... Args>
struct is_const_member_function<R (C::*)(Args...) const> : cuda::std::true_type
{};

template <typename R, typename C, typename... Args>
struct is_const_member_function<R (C::*)(Args...) const noexcept> : cuda::std::true_type
{};

template <typename T>
constexpr bool is_const_member_function_v = is_const_member_function<T>::value;

//----------------------------------------------------------------------------------------------------------------------
// mask utilities

struct is_even
{
  template <typename I>
  TEST_FUNC constexpr bool operator()(I i) const noexcept
  {
    return i % 2 == 0;
  }
};

struct is_first_half
{
  template <typename I>
  TEST_FUNC constexpr bool operator()(I i) const noexcept
  {
    return i < 2;
  }
};

template <int Val>
struct is_index
{
  template <typename I>
  TEST_FUNC constexpr bool operator()(I i) const noexcept
  {
    return i == Val;
  }
};

template <int Val>
struct is_greater_equal_than_index
{
  template <typename I>
  TEST_FUNC constexpr bool operator()(I i) const noexcept
  {
    return i >= Val;
  }
};

template <int Val>
struct is_less_than_index
{
  template <typename I>
  TEST_FUNC constexpr bool operator()(I i) const noexcept
  {
    return i < Val;
  }
};

template <int Bytes>
using integer_from_t = cuda::std::__make_nbit_int_t<Bytes * 8, true>;

//----------------------------------------------------------------------------------------------------------------------
// vec utilities

template <typename T>
struct iota_generator
{
  template <typename I>
  TEST_FUNC constexpr T operator()(I i) const noexcept
  {
    return static_cast<T>(i + 1);
  }
};

template <typename T, int N>
TEST_FUNC constexpr simd::basic_vec<T, simd::fixed_size<N>> make_iota_vec()
{
  cuda::std::array<T, N> arr{};
  for (int i = 0; i < N; ++i)
  {
    arr[i] = static_cast<T>(i);
  }
  return simd::basic_vec<T, simd::fixed_size<N>>(arr);
}

//----------------------------------------------------------------------------------------------------------------------
// math utilities

template <typename T>
struct math_values
{
  template <typename I>
  TEST_FUNC constexpr T operator()(I i) const noexcept
  {
    return static_cast<T>(static_cast<int>(i) - 1) / T{4};
  }
};

template <typename T>
struct positive_math_values
{
  template <typename I>
  TEST_FUNC constexpr T operator()(I i) const noexcept
  {
    return static_cast<T>(i + 1) / T{4};
  }
};

template <typename T>
TEST_FUNC bool almost_equal(T lhs, T rhs, T tolerance) noexcept
{
  return cuda::std::fabs(lhs - rhs) <= tolerance;
}

// Each vec test file must define test_type<T, N>() and then define test() using this macro.
// clang-format off
#if defined(__cccl_lib_char8_t)
#  define _SIMD_TEST_CHAR8_T()                                    \
    test_type<char8_t, 1>();                                      \
    test_type<char8_t, 4>();
#else
#  define _SIMD_TEST_CHAR8_T()
#endif

#if _CCCL_HAS_INT128()
#  define _SIMD_TEST_INT128()                                     \
    test_type<__int128_t, 1>();                                   \
    test_type<__int128_t, 4>();
#else
#  define _SIMD_TEST_INT128()
#endif

#if _LIBCUDACXX_HAS_NVFP16()
#  define _SIMD_TEST_FP16(_N)                                     \
    test_type<__half, _N>();
#else
#  define _SIMD_TEST_FP16(_N)
#endif

#if _LIBCUDACXX_HAS_NVBF16()
#  define _SIMD_TEST_BF16(_N)                                     \
    test_type<__nv_bfloat16, _N>();
#else
#  define _SIMD_TEST_BF16(_N)
#endif

// Each simd.math test file must define test_type<T, N>() and then define test() using this macro.
#define DEFINE_SIMD_MATH_FLOATING_TEST()                           \
  TEST_FUNC bool test()                                            \
  {                                                                \
    test_type<float, 4>();                                         \
    test_type<double, 4>();                                        \
    return true;                                                   \
  }

// __half and __nv_bfloat16 constructors are not constexpr, they are tested only at runtime via test_runtime().
#define DEFINE_SIMD_MATH_FLOATING_TEST_RUNTIME()                  \
  TEST_FUNC bool test_runtime()                                   \
  {                                                               \
    _SIMD_TEST_FP16(4)                                            \
    _SIMD_TEST_BF16(4)                                            \
    return true;                                                  \
  }

#define DEFINE_BASIC_VEC_TEST_RUNTIME()                           \
  TEST_FUNC bool test_runtime()                                   \
  {                                                               \
    _SIMD_TEST_FP16(1)                                            \
    _SIMD_TEST_FP16(4)                                            \
    _SIMD_TEST_BF16(1)                                            \
    _SIMD_TEST_BF16(4)                                            \
    return true;                                                  \
  }

#define DEFINE_BASIC_VEC_TEST()                                   \
  TEST_FUNC constexpr bool test()                       \
  {                                                               \
    test_type<int8_t, 1>();                                       \
    test_type<int8_t, 4>();                                       \
    test_type<int16_t, 1>();                                      \
    test_type<int16_t, 4>();                                      \
    test_type<int32_t, 1>();                                      \
    test_type<int32_t, 4>();                                      \
    test_type<int64_t, 1>();                                      \
    test_type<int64_t, 4>();                                      \
    test_type<uint8_t, 1>();                                      \
    test_type<uint8_t, 4>();                                      \
    test_type<uint16_t, 1>();                                     \
    test_type<uint16_t, 4>();                                     \
    test_type<uint32_t, 1>();                                     \
    test_type<uint32_t, 4>();                                     \
    test_type<uint64_t, 1>();                                     \
    test_type<uint64_t, 4>();                                     \
    test_type<char16_t, 1>();                                     \
    test_type<char16_t, 4>();                                     \
    test_type<char32_t, 1>();                                     \
    test_type<char32_t, 4>();                                     \
    test_type<wchar_t, 1>();                                      \
    test_type<wchar_t, 4>();                                      \
    _SIMD_TEST_CHAR8_T()                                          \
    test_type<float, 1>();                                        \
    test_type<float, 4>();                                        \
    test_type<double, 1>();                                       \
    test_type<double, 4>();                                       \
    _SIMD_TEST_INT128()                                           \
    return true;                                                  \
  }
// clang-format on

#endif // SIMD_TEST_UTILS_H
