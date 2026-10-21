//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef TEST_CUDA_ITERATOR_COUNTING_ITERATOR_H
#define TEST_CUDA_ITERATOR_COUNTING_ITERATOR_H

#include <cuda/std/cassert>
#include <cuda/std/iterator>

#include "test_macros.h"

struct PlusOne
{
  TEST_FUNC constexpr int operator()(int x) const noexcept
  {
    return x + 1;
  }
};

struct PlusOneMutable
{
  TEST_FUNC constexpr int operator()(int x) noexcept
  {
    return x + 1;
  }
};

struct PlusOneMayThrow
{
  TEST_FUNC constexpr int operator()(int x) const
  {
    return x + 1;
  }
};

#if !TEST_COMPILER(NVRTC)
struct PlusOneHost
{
  constexpr PlusOneHost() noexcept {}
  constexpr int operator()(int x) const noexcept
  {
    return x + 1;
  }
};
#endif // !TEST_COMPILER(NVRTC)

#if _CCCL_CUDA_COMPILATION()
struct PlusOneDevice
{
  TEST_DEVICE_FUNC constexpr PlusOneDevice() noexcept {}
  TEST_DEVICE_FUNC constexpr int operator()(int x) const noexcept
  {
    return x + 1;
  }
};
#endif // _CCCL_CUDA_COMPILATION()

struct NotDefaultConstructiblePlusOne
{
  TEST_FUNC constexpr NotDefaultConstructiblePlusOne(int) noexcept {}
  TEST_FUNC constexpr int operator()(int x) const noexcept
  {
    return x + 1;
  }
};

struct TimesTwo
{
  TEST_FUNC constexpr int operator()(int x) const noexcept
  {
    return x * 2;
  }
};

struct TimesTwoMayThrow
{
  TEST_FUNC constexpr int operator()(int x) const
  {
    return x * 2;
  }
};

#endif // TEST_CUDA_ITERATOR_COUNTING_ITERATOR_H
