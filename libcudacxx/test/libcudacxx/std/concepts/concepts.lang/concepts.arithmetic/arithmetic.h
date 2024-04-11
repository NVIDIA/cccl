//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#ifndef LIBCXX_TEST_CONCEPTS_LANG_CONCEPTS_ARITHMETIC_H_
#define LIBCXX_TEST_CONCEPTS_LANG_CONCEPTS_ARITHMETIC_H_

#include <cuda/std/concepts>

#include "test_macros.h"

#if TEST_STD_VER > 2017
// This overload should never be called. It exists solely to force subsumption.
template <cuda::std::integral I>
__host__ __device__ constexpr bool CheckSubsumption(I)
{
  return false;
}

template <cuda::std::integral I>
  requires cuda::std::signed_integral<I> && (!cuda::std::unsigned_integral<I>)
__host__ __device__ constexpr bool CheckSubsumption(I)
{
  return cuda::std::is_signed_v<I>;
}

template <cuda::std::integral I>
  requires cuda::std::unsigned_integral<I> && (!cuda::std::signed_integral<I>)
__host__ __device__ constexpr bool CheckSubsumption(I)
{
  return cuda::std::is_unsigned_v<I>;
}
#endif // TEST_STD_VER > 2017

enum ClassicEnum
{
  a,
  b,
  c
};
enum class ScopedEnum
{
  x,
  y,
  z
};
struct EmptyStruct
{};

#endif // LIBCXX_TEST_CONCEPTS_LANG_CONCEPTS_ARITHMETIC_H_
