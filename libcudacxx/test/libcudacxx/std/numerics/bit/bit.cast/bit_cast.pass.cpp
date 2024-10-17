//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/bit>
//
// template<class To, class From>
//   constexpr To bit_cast(const From& from) noexcept;

#include <cuda/std/array>
#include <cuda/std/bit>
#include <cuda/std/cassert>
#include <cuda/std/cmath>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

#include "test_macros.h"

__host__ __device__ cuda::std::size_t test_memcmp(void* lhs, void* rhs, size_t bytes) noexcept
{
  const unsigned char* clhs = (const unsigned char*) lhs;
  const unsigned char* crhs = (const unsigned char*) rhs;

  for (; bytes > 0; --bytes)
  {
    if (*clhs++ != *crhs++)
    {
      return clhs[-1] < crhs[-1] ? -1 : 1;
    }
  }
  return 0;
}

// cuda::std::bit_cast does not preserve padding bits, so if T has padding bits,
// the results might not memcmp cleanly.
template <bool HasUniqueObjectRepresentations = true, typename T>
__host__ __device__ void test_roundtrip_through_buffer(T from)
{
  struct Buffer
  {
    char buffer[sizeof(T)];
  };
  Buffer middle  = cuda::std::bit_cast<Buffer>(from);
  T to           = cuda::std::bit_cast<T>(middle);
  Buffer middle2 = cuda::std::bit_cast<Buffer>(to);

  assert((from == to) == (from == from)); // because NaN

  _CCCL_IF_CONSTEXPR (HasUniqueObjectRepresentations)
  {
    assert(test_memcmp(&from, &middle, sizeof(T)) == 0);
    assert(test_memcmp(&to, &middle, sizeof(T)) == 0);
    assert(test_memcmp(&middle, &middle2, sizeof(T)) == 0);
  }
}

template <bool HasUniqueObjectRepresentations = true, typename T>
__host__ __device__ void test_roundtrip_through_nested_T(T from)
{
  struct Nested
  {
    T x;
  };
  static_assert(sizeof(Nested) == sizeof(T), "");

  Nested middle  = cuda::std::bit_cast<Nested>(from);
  T to           = cuda::std::bit_cast<T>(middle);
  Nested middle2 = cuda::std::bit_cast<Nested>(to);

  assert((from == to) == (from == from)); // because NaN

  _CCCL_IF_CONSTEXPR (HasUniqueObjectRepresentations)
  {
    assert(test_memcmp(&from, &middle, sizeof(T)) == 0);
    assert(test_memcmp(&to, &middle, sizeof(T)) == 0);
    assert(test_memcmp(&middle, &middle2, sizeof(T)) == 0);
  }
}

template <typename Intermediate, bool HasUniqueObjectRepresentations = true, typename T>
__host__ __device__ void test_roundtrip_through(T from)
{
  static_assert(sizeof(Intermediate) == sizeof(T), "");

  Intermediate middle  = cuda::std::bit_cast<Intermediate>(from);
  T to                 = cuda::std::bit_cast<T>(middle);
  Intermediate middle2 = cuda::std::bit_cast<Intermediate>(to);

  assert((from == to) == (from == from)); // because NaN

  _CCCL_IF_CONSTEXPR (HasUniqueObjectRepresentations)
  {
    assert(test_memcmp(&from, &middle, sizeof(T)) == 0);
    assert(test_memcmp(&to, &middle, sizeof(T)) == 0);
    assert(test_memcmp(&middle, &middle2, sizeof(T)) == 0);
  }
}

template <typename T>
__host__ __device__ _LIBCUDACXX_CONSTEXPR_BIT_CAST cuda::std::array<T, 10> generate_signed_integral_values()
{
  return {cuda::std::numeric_limits<T>::min(),
          cuda::std::numeric_limits<T>::min() + 1,
          static_cast<T>(-2),
          static_cast<T>(-1),
          static_cast<T>(0),
          static_cast<T>(1),
          static_cast<T>(2),
          static_cast<T>(3),
          cuda::std::numeric_limits<T>::max() - 1,
          cuda::std::numeric_limits<T>::max()};
}

template <typename T>
__host__ __device__ _LIBCUDACXX_CONSTEXPR_BIT_CAST cuda::std::array<T, 6> generate_unsigned_integral_values()
{
  return {static_cast<T>(0),
          static_cast<T>(1),
          static_cast<T>(2),
          static_cast<T>(3),
          static_cast<T>(cuda::std::numeric_limits<T>::max() - 1),
          cuda::std::numeric_limits<T>::max()};
}

__host__ __device__ bool tests()
{
  for (bool b : {false, true})
  {
    test_roundtrip_through_nested_T(b);
    test_roundtrip_through_buffer(b);
    test_roundtrip_through<char>(b);
  }

  for (char c : {'\0', 'a', 'b', 'c', 'd'})
  {
    test_roundtrip_through_nested_T(c);
    test_roundtrip_through_buffer(c);
  }

  // Fundamental signed integer types
  for (signed char i : generate_signed_integral_values<signed char>())
  {
    test_roundtrip_through_nested_T(i);
    test_roundtrip_through_buffer(i);
  }

  for (short i : generate_signed_integral_values<short>())
  {
    test_roundtrip_through_nested_T(i);
    test_roundtrip_through_buffer(i);
  }

  for (int i : generate_signed_integral_values<int>())
  {
    test_roundtrip_through_nested_T(i);
    test_roundtrip_through_buffer(i);
    test_roundtrip_through<float>(i);
  }

  for (long i : generate_signed_integral_values<long>())
  {
    test_roundtrip_through_nested_T(i);
    test_roundtrip_through_buffer(i);
  }

  for (long long i : generate_signed_integral_values<long long>())
  {
    test_roundtrip_through_nested_T(i);
    test_roundtrip_through_buffer(i);
    test_roundtrip_through<double>(i);
  }

  // Fundamental unsigned integer types
  for (unsigned char i : generate_unsigned_integral_values<unsigned char>())
  {
    test_roundtrip_through_nested_T(i);
    test_roundtrip_through_buffer(i);
  }

  for (unsigned short i : generate_unsigned_integral_values<unsigned short>())
  {
    test_roundtrip_through_nested_T(i);
    test_roundtrip_through_buffer(i);
  }

  for (unsigned int i : generate_unsigned_integral_values<unsigned int>())
  {
    test_roundtrip_through_nested_T(i);
    test_roundtrip_through_buffer(i);
    test_roundtrip_through<float>(i);
  }

  for (unsigned long i : generate_unsigned_integral_values<unsigned long>())
  {
    test_roundtrip_through_nested_T(i);
    test_roundtrip_through_buffer(i);
  }

  for (unsigned long long i : generate_unsigned_integral_values<unsigned long long>())
  {
    test_roundtrip_through_nested_T(i);
    test_roundtrip_through_buffer(i);
    test_roundtrip_through<double>(i);
  }

  // Fixed width signed integer types
  for (cuda::std::int32_t i : generate_signed_integral_values<cuda::std::int32_t>())
  {
    test_roundtrip_through_nested_T(i);
    test_roundtrip_through_buffer(i);
    test_roundtrip_through<int>(i);
    test_roundtrip_through<cuda::std::uint32_t>(i);
    test_roundtrip_through<float>(i);
  }

  for (cuda::std::int64_t i : generate_signed_integral_values<cuda::std::int64_t>())
  {
    test_roundtrip_through_nested_T(i);
    test_roundtrip_through_buffer(i);
    test_roundtrip_through<long long>(i);
    test_roundtrip_through<cuda::std::uint64_t>(i);
    test_roundtrip_through<double>(i);
  }

  // Fixed width unsigned integer types
  for (cuda::std::uint32_t i : generate_unsigned_integral_values<cuda::std::uint32_t>())
  {
    test_roundtrip_through_nested_T(i);
    test_roundtrip_through_buffer(i);
    test_roundtrip_through<int>(i);
    test_roundtrip_through<cuda::std::int32_t>(i);
    test_roundtrip_through<float>(i);
  }

  for (cuda::std::uint64_t i : generate_unsigned_integral_values<cuda::std::uint64_t>())
  {
    test_roundtrip_through_nested_T(i);
    test_roundtrip_through_buffer(i);
    test_roundtrip_through<long long>(i);
    test_roundtrip_through<cuda::std::int64_t>(i);
    test_roundtrip_through<double>(i);
  }

  // Floating point types
  for (float i :
       {0.0f,
        1.0f,
        -1.0f,
        10.0f,
        -10.0f,
        1e10f,
        1e-10f,
        1e20f,
        1e-20f,
        2.71828f,
        3.14159f,
#if !defined(TEST_COMPILER_NVRTC) && !defined(TEST_COMPILER_CLANG_CUDA)
        cuda::std::nanf(""),
#endif // !TEST_COMPILER_NVRTC && !TEST_COMPILER_CLANG_CUDA
        __builtin_nanf("0x55550001"), // NaN with a payload
        cuda::std::numeric_limits<float>::signaling_NaN(),
        cuda::std::numeric_limits<float>::quiet_NaN(),
        cuda::std::numeric_limits<float>::infinity()})
  {
    test_roundtrip_through_nested_T(i);
    test_roundtrip_through_buffer(i);
    test_roundtrip_through<int>(i);
  }

  for (double i :
       {0.0,
        1.0,
        -1.0,
        10.0,
        -10.0,
        1e10,
        1e-10,
        1e100,
        1e-100,
        2.718281828459045,
        3.141592653589793238462643383279502884197169399375105820974944,
#if !defined(TEST_COMPILER_NVRTC) && !defined(TEST_COMPILER_CLANG_CUDA)
        cuda::std::nan(""),
#endif // !TEST_COMPILER_NVRTC && !TEST_COMPILER_CLANG_CUDA
        cuda::std::numeric_limits<double>::signaling_NaN(),
        cuda::std::numeric_limits<double>::quiet_NaN(),
        cuda::std::numeric_limits<double>::infinity()})
  {
    test_roundtrip_through_nested_T(i);
    test_roundtrip_through_buffer(i);
    test_roundtrip_through<long long>(i);
  }

#ifdef _LIBCUDACXX_HAS_NVFP16
  // Extended floating point type __half
  for (__half i :
       {__float2half(0.0f),
        __float2half(1.0f),
        __float2half(-1.0f),
        __float2half(10.0f),
        __float2half(-10.0f),
        __float2half(2.71828f),
        __float2half(3.14159f)})
  {
    test_roundtrip_through_nested_T(i);
    test_roundtrip_through_buffer(i);
    test_roundtrip_through<cuda::std::int16_t>(i);
  }
#endif // _LIBCUDACXX_HAS_NVFP16

#ifdef _LIBCUDACXX_HAS_NVBF16
  // Extended floating point type __half
  for (__nv_bfloat16 i :
       {__float2bfloat16(0.0f),
        __float2bfloat16(1.0f),
        __float2bfloat16(-1.0f),
        __float2bfloat16(10.0f),
        __float2bfloat16(-10.0f),
        __float2bfloat16(2.71828f),
        __float2bfloat16(3.14159f)})
  {
    test_roundtrip_through_nested_T(i);
    test_roundtrip_through_buffer(i);
    test_roundtrip_through<cuda::std::int16_t>(i);
  }
#endif // _LIBCUDACXX_HAS_NVBF16

  // Test pointers
  {
    {
      int obj = 3;
      void* p = &obj;
      test_roundtrip_through_nested_T(p);
      test_roundtrip_through_buffer(p);
      test_roundtrip_through<void*>(p);
      test_roundtrip_through<char*>(p);
      test_roundtrip_through<int*>(p);
    }
    {
      int obj = 3;
      int* p  = &obj;
      test_roundtrip_through_nested_T(p);
      test_roundtrip_through_buffer(p);
      test_roundtrip_through<int*>(p);
      test_roundtrip_through<char*>(p);
      test_roundtrip_through<void*>(p);
    }
  }

  return true;
}

#if defined(_CCCL_BUILTIN_BIT_CAST)
__host__ __device__ constexpr bool basic_constexpr_test()
{
#  if TEST_STD_VER >= 2014
  struct Nested
  {
    char buffer[sizeof(int)];
  };
  int from      = 3;
  Nested middle = cuda::std::bit_cast<Nested>(from);
  int to        = cuda::std::bit_cast<int>(middle);
  return (from == to);
#  else // ^^^ C++14 ^^^ / vvv C++11 vvv
  // only do a sanity check in C++11
  return (3 == cuda::std::bit_cast<unsigned>(3u));
#  endif // TEST_STD_VER >= 2014
}
#endif // _CCCL_BUILTIN_BIT_CAST

int main(int, char**)
{
  tests();
#if defined(_CCCL_BUILTIN_BIT_CAST)
  static_assert(basic_constexpr_test(), "");
#endif // _CCCL_BUILTIN_BIT_CAST
  return 0;
}
