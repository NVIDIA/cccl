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
#include <cuda/std/complex>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

#include "fp_compare.h"
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
// Approximate floating-point comparison for SIMD complex math tests

// even if SIMD applies mathematical operations for each component, the compilers could still perform different
// optimizations between library and test code. nvc++ and clang especially produce slightly different results for the
// same input.
TEST_FUNC inline void is_fp_close_runtime(float a, float b)
{
  assert(fptest_close_pct(a, b, 1.e-4f));
}

TEST_FUNC inline void is_fp_close_runtime(double a, double b)
{
  assert(fptest_close_pct(a, b, 1.e-12));
}

#if _LIBCUDACXX_HAS_NVFP16()
TEST_FUNC inline void is_fp_close_runtime(__half a, __half b)
{
  assert(fptest_close_pct(static_cast<float>(a), static_cast<float>(b), 1.e-1f));
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
TEST_FUNC inline void is_fp_close_runtime(__nv_bfloat16 a, __nv_bfloat16 b)
{
  assert(fptest_close_pct(static_cast<float>(a), static_cast<float>(b), 5.e-1f));
}
#endif // _LIBCUDACXX_HAS_NVBF16()

template <typename T>
TEST_FUNC void is_fp_close_runtime(const cuda::std::complex<T>& a, const cuda::std::complex<T>& b)
{
  is_fp_close_runtime(a.real(), b.real());
  is_fp_close_runtime(a.imag(), b.imag());
}

// compile-time tests are bitwise identical
template <typename T>
TEST_FUNC constexpr void is_fp_close(const T& a, const T& b)
{
  if (cuda::std::__cccl_default_is_constant_evaluated())
  {
    assert(a == b);
  }
  else
  {
    is_fp_close_runtime(a, b);
  }
}

//----------------------------------------------------------------------------------------------------------------------
// vec utilities

template <typename T, int Offset = 1>
struct iota_generator
{
  template <typename I>
  TEST_FUNC constexpr T operator()(I i) const noexcept
  {
    return static_cast<T>(i + Offset);
  }
};

template <typename T, int Offset>
struct offset_generator
{
  template <typename I>
  TEST_FUNC constexpr T operator()(I i) const noexcept
  {
    return static_cast<T>(i + Offset);
  }
};

template <typename T, int RealOffset, int ImagOffset>
struct complex_generator
{
  template <typename I>
  TEST_FUNC constexpr cuda::std::complex<T> operator()(I i) const noexcept
  {
    return cuda::std::complex<T>(static_cast<T>(i + RealOffset), static_cast<T>(i + ImagOffset));
  }
};

// Four complex values spanning all quadrants with diverse magnitudes; shared across complex tests.
template <typename T>
struct complex_diverse_generator
{
  template <typename I>
  TEST_FUNC constexpr cuda::std::complex<T> operator()(I i) const noexcept
  {
    switch (static_cast<int>(i) & 3)
    {
      case 0:
        return cuda::std::complex<T>(T(1.5), T(0.4));
      case 1:
        return cuda::std::complex<T>(T(-0.7), T(1.2));
      case 2:
        return cuda::std::complex<T>(T(0.8), T(-1.3));
      default:
        return cuda::std::complex<T>(T(-1.1), T(-0.6));
    }
  }
};

template <typename T, int N, int Offset = 1>
TEST_FUNC constexpr cuda::std::array<T, N> make_iota_array()
{
  cuda::std::array<T, N> arr{};
  for (int i = 0; i < N; ++i)
  {
    arr[i] = static_cast<T>(i + Offset);
  }
  return arr;
}

template <typename T, int N>
TEST_FUNC constexpr cuda::std::array<T, N> make_iota_array(int offset)
{
  cuda::std::array<T, N> arr{};
  for (int i = 0; i < N; ++i)
  {
    arr[i] = static_cast<T>(i + offset);
  }
  return arr;
}

template <typename T, int N, int Offset = 0>
TEST_FUNC constexpr cuda::std::array<T, N> make_reverse_iota_array()
{
  cuda::std::array<T, N> arr{};
  for (int i = 0; i < N; ++i)
  {
    arr[i] = static_cast<T>(N - 1 - i + Offset);
  }
  return arr;
}

// Elementwise comparison of a basic_vec against a cuda::std::array
template <typename T, typename Abi, typename U, size_t N>
TEST_FUNC constexpr bool operator==(const simd::basic_vec<T, Abi>& vec, const cuda::std::array<U, N>& arr)
{
  static_assert(simd::basic_vec<T, Abi>::size() == static_cast<int>(N));
  for (size_t i = 0; i < N; ++i)
  {
    if (vec[i] != arr[i])
    {
      return false;
    }
  }
  return true;
}

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

template <typename T, int N>
TEST_FUNC constexpr simd::basic_vec<T, simd::fixed_size<N>> make_iota_reverse_vec()
{
  cuda::std::array<T, N> arr{};
  for (int i = 0; i < N; ++i)
  {
    arr[i] = static_cast<T>(N - 1 - i);
  }
  return simd::basic_vec<T, simd::fixed_size<N>>(arr);
}
//----------------------------------------------------------------------------------------------------------------------
// bit utilities

template <typename T>
struct bit_values
{
  template <typename I>
  TEST_FUNC constexpr T operator()(I) const noexcept
  {
    return static_cast<T>((I::value + 1) * 3);
  }
};

// Each simd.bit test file must define test_constraints() and a test functor template
// clang-format off
#define _SIMD_BIT_TEST_SIGNED_TYPES(_Test)                         \
  _Test<int8_t, 1>{}();                                            \
  _Test<int8_t, 4>{}();                                            \
  _Test<int16_t, 1>{}();                                           \
  _Test<int16_t, 4>{}();                                           \
  _Test<int32_t, 1>{}();                                           \
  _Test<int32_t, 4>{}();                                           \
  _Test<int64_t, 1>{}();                                           \
  _Test<int64_t, 4>{}();

#define _SIMD_BIT_TEST_UNSIGNED_TYPES(_Test)                       \
  _Test<uint8_t, 1>{}();                                           \
  _Test<uint8_t, 4>{}();                                           \
  _Test<uint16_t, 1>{}();                                          \
  _Test<uint16_t, 4>{}();                                          \
  _Test<uint32_t, 1>{}();                                          \
  _Test<uint32_t, 4>{}();                                          \
  _Test<uint64_t, 1>{}();                                          \
  _Test<uint64_t, 4>{}();

#define DEFINE_SIMD_BIT_INTEGRAL_TEST(_Test)                      \
  TEST_FUNC constexpr bool test()                                 \
  {                                                               \
    _SIMD_BIT_TEST_SIGNED_TYPES(_Test)                            \
    _SIMD_BIT_TEST_UNSIGNED_TYPES(_Test)                          \
    test_constraints();                                           \
    return true;                                                  \
  }

#define DEFINE_SIMD_BIT_UNSIGNED_TEST(_Test)                      \
  TEST_FUNC constexpr bool test()                                 \
  {                                                               \
    _SIMD_BIT_TEST_UNSIGNED_TYPES(_Test)                          \
    test_constraints();                                           \
    return true;                                                  \
  }
// clang-format on

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
#  define _SIMD_TEST_FP16()                                       \
    test_type<__half, 1>();                                       \
    test_type<__half, 4>();
#else
#  define _SIMD_TEST_FP16()
#endif

#if _LIBCUDACXX_HAS_NVBF16()
#  define _SIMD_TEST_BF16()                                       \
    test_type<__nv_bfloat16, 1>();                                \
    test_type<__nv_bfloat16, 4>();
#else
#  define _SIMD_TEST_BF16()
#endif

// __half and __nv_bfloat16 constructors are not constexpr (CUDA toolkit limitation),
// so they are tested only at runtime via test_runtime().
#define DEFINE_BASIC_VEC_TEST_RUNTIME()                           \
  TEST_FUNC bool test_runtime()                                   \
  {                                                               \
    _SIMD_TEST_FP16()                                             \
    _SIMD_TEST_BF16()                                             \
    return true;                                                  \
  }

#define DEFINE_BASIC_VEC_TEST()                                   \
  TEST_FUNC constexpr bool test()                                 \
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
