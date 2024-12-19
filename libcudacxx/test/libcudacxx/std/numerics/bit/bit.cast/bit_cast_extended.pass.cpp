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
#include <cuda/std/tuple>
#include <cuda/std/utility>

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

template <class T>
__host__ __device__ bool equal(T& lhs, T& rhs)
{
  return test_memcmp(&lhs, &rhs, sizeof(T)) == 0;
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

  assert(equal(from, to) == equal(from, from)); // because NaN

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

#define REPEAT_1(base_type, index) base_type(input[index][0])
#define REPEAT_2(base_type, index) REPEAT_1(base_type, index), base_type(input[index][1])
#define REPEAT_3(base_type, index) REPEAT_2(base_type, index), base_type(input[index][2])
#define REPEAT_4(base_type, index) REPEAT_3(base_type, index), base_type(input[index][3])

#define TEST_CUDA_VECTOR_TYPE(base_type, size)           \
  {                                                      \
    for (base_type##size i :                             \
         {base_type##size{REPEAT_##size(base_type, 0)},  \
          base_type##size{REPEAT_##size(base_type, 1)},  \
          base_type##size{REPEAT_##size(base_type, 2)},  \
          base_type##size{REPEAT_##size(base_type, 3)},  \
          base_type##size{REPEAT_##size(base_type, 4)},  \
          base_type##size{REPEAT_##size(base_type, 5)},  \
          base_type##size{REPEAT_##size(base_type, 6)}}) \
    {                                                    \
      test_roundtrip_through_buffer(i);                  \
    }                                                    \
  }

#define TEST_CUDA_VECTOR_TYPES(base_type) \
  TEST_CUDA_VECTOR_TYPE(base_type, 1)     \
  TEST_CUDA_VECTOR_TYPE(base_type, 2)     \
  TEST_CUDA_VECTOR_TYPE(base_type, 3)     \
  TEST_CUDA_VECTOR_TYPE(base_type, 4)

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
  using pair = cuda::std::pair<float, int>;
  for (pair i :
       {pair{0.0f, 1},
        pair{1.0f, 2},
        pair{-1.0f, 3},
        pair{10.0f, 4},
        pair{-10.0f, 5},
        pair{2.71828f, 6},
        pair{3.14159f, 7}})
  {
    test_roundtrip_through_buffer(i);
  }

#if defined(_CCCL_BUILTIN_BIT_CAST) // tuple is not trivially default constructible
  using tuple = cuda::std::tuple<float, int, short>;
  for (tuple i :
       {tuple{0.0f, 1, -1},
        tuple{1.0f, 2, -2},
        tuple{-1.0f, 3, -3},
        tuple{10.0f, 4, -4},
        tuple{-10.0f, 5, -5},
        tuple{2.71828f, 6, -6},
        tuple{3.14159f, 7, -7}})
  {
    test_roundtrip_through_buffer(i);
  }
#endif // _CCCL_BUILTIN_BIT_CAST

  using array = cuda::std::array<float, 2>;
  for (array i :
       {array{0.0f, 1.0f},
        array{1.0f, 2.0f},
        array{-1.0f, 3.0f},
        array{10.0f, 4.0f},
        array{-10.0f, 5.0f},
        array{2.71828f, 6.0f},
        array{3.14159f, 7.0f}})
  {
    test_roundtrip_through_buffer(i);
  }

  float carray[2] = {0.0f, 1.0f};
  test_roundtrip_through_buffer(carray);

  // test cuda vector types except __half2 and __nv_bfloat162 because they are cursed
  constexpr double input[7][4] = {
    {0.0, 1.0, -7.0, -0.0},
    {1.0, 2.0, -7.0, -1.0},
    {-1.0, 3.0, -7.0, 1.0},
    {10.0, 4.0, -7.0, -10.0},
    {-10.0, 5.0, -7.0, 10.0},
    {2.71828, 6.0, -7.0, -2.71828},
    {3.14159, 7.0, -7.0, -3.14159}};

#if !_CCCL_CUDA_COMPILER(CLANG)
  using uchar  = unsigned char;
  using ushort = unsigned short;
  using uint   = unsigned int;
  using ulong  = unsigned long;
#endif // !_CCCL_CUDA_COMPILER(CLANG)
  using longlong  = long long;
  using ulonglong = unsigned long long;

  TEST_CUDA_VECTOR_TYPES(char)
  TEST_CUDA_VECTOR_TYPES(uchar)
  TEST_CUDA_VECTOR_TYPES(short)
  TEST_CUDA_VECTOR_TYPES(ushort)
  TEST_CUDA_VECTOR_TYPES(int)
  TEST_CUDA_VECTOR_TYPES(uint)
  TEST_CUDA_VECTOR_TYPES(long)
  TEST_CUDA_VECTOR_TYPES(ulong)
  TEST_CUDA_VECTOR_TYPES(longlong)
  TEST_CUDA_VECTOR_TYPES(ulonglong)
  TEST_CUDA_VECTOR_TYPES(float)
  TEST_CUDA_VECTOR_TYPES(double)

  using dim = unsigned int;
  TEST_CUDA_VECTOR_TYPE(dim, 3)

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
    test_roundtrip_through_buffer(i);
  }
#endif // _LIBCUDACXX_HAS_NVFP16

#ifdef _LIBCUDACXX_HAS_NVBF16
  // Extended floating point type __nv_bfloat16
  for (__nv_bfloat16 i :
       {__float2bfloat16(0.0f),
        __float2bfloat16(1.0f),
        __float2bfloat16(-1.0f),
        __float2bfloat16(10.0f),
        __float2bfloat16(-10.0f),
        __float2bfloat16(2.71828f),
        __float2bfloat16(3.14159f)})
  {
    test_roundtrip_through_buffer(i);
  }
#endif // _LIBCUDACXX_HAS_NVBF16

  return true;
}

int main(int, char**)
{
  tests();
  return 0;
}
