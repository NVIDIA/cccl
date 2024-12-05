//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// #include <memory>

// template<size_t N, class T>
// [[nodiscard]] constexpr T* assume_aligned(T* ptr);

#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/memory>

#include "test_macros.h"

#if defined(TEST_COMPILER_MSVC)
#  pragma warning(disable : 4324) // structure was padded due to alignment specifier
#endif // TEST_COMPILER_MSVC

template <typename T>
__host__ __device__ TEST_CONSTEXPR_CXX14 void check(T* p)
{
  ASSERT_SAME_TYPE(T*, decltype(cuda::std::assume_aligned<1>(p)));
  constexpr cuda::std::size_t alignment = alignof(T);

  _CCCL_IF_CONSTEXPR (alignment >= 1)
  {
    assert(p == cuda::std::assume_aligned<1>(p));
  }
  _CCCL_IF_CONSTEXPR (alignment >= 2)
  {
    assert(p == cuda::std::assume_aligned<2>(p));
  }
  _CCCL_IF_CONSTEXPR (alignment >= 4)
  {
    assert(p == cuda::std::assume_aligned<4>(p));
  }
  _CCCL_IF_CONSTEXPR (alignment >= 8)
  {
    assert(p == cuda::std::assume_aligned<8>(p));
  }
  _CCCL_IF_CONSTEXPR (alignment >= 16)
  {
    assert(p == cuda::std::assume_aligned<16>(p));
  }
  _CCCL_IF_CONSTEXPR (alignment >= 32)
  {
    assert(p == cuda::std::assume_aligned<32>(p));
  }
  _CCCL_IF_CONSTEXPR (alignment >= 64)
  {
    assert(p == cuda::std::assume_aligned<64>(p));
  }
  _CCCL_IF_CONSTEXPR (alignment >= 128)
  {
    assert(p == cuda::std::assume_aligned<128>(p));
  }
}

struct S
{};
struct alignas(4) S4
{};
struct alignas(8) S8
{};
struct alignas(16) S16
{};
struct alignas(32) S32
{};
struct alignas(64) S64
{};
struct alignas(128) S128
{};

__host__ __device__ TEST_CONSTEXPR_CXX14 bool tests()
{
  char c{};
  int i{};
  long l{};
  double d{};
  check(&c);
  check(&i);
  check(&l);
  check(&d);

  S s{};
  S4 s4{};
  S8 s8{};
  S16 s16{};
  S32 s32{};
  S64 s64{};
  S128 s128{};
  check(&s);
  check(&s4);
  check(&s8);
  check(&s16);
  check(&s32);
  check(&s64);
  check(&s128);

  return true;
}

int main(int, char**)
{
  tests();
#if TEST_STD_VER >= 2014
  static_assert(tests(), "");
#endif // TEST_STD_VER >= 2014

  return 0;
}
