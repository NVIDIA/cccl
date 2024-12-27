//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/ptx>

#include <cuda/ptx>
#include <cuda/std/cstdint>

#include "test_macros.h"

template <class Result, class Arg, class MulMode>
__device__ void test_mul(MulMode mode)
{
  auto res = cuda::ptx::mul(mode, Arg{}, Arg{});
  ASSERT_SAME_TYPE(Result, decltype(res));
}

__global__ void test_global_kernel()
{
  test_mul<cuda::std::int16_t, cuda::std::int16_t>(cuda::ptx::mul_mode_lo);
  test_mul<cuda::std::uint16_t, cuda::std::uint16_t>(cuda::ptx::mul_mode_lo);
  test_mul<cuda::std::int32_t, cuda::std::int32_t>(cuda::ptx::mul_mode_lo);
  test_mul<cuda::std::uint32_t, cuda::std::uint32_t>(cuda::ptx::mul_mode_lo);
  test_mul<cuda::std::int64_t, cuda::std::int64_t>(cuda::ptx::mul_mode_lo);
  test_mul<cuda::std::uint64_t, cuda::std::uint64_t>(cuda::ptx::mul_mode_lo);

  test_mul<cuda::std::int16_t, cuda::std::int16_t>(cuda::ptx::mul_mode_hi);
  test_mul<cuda::std::uint16_t, cuda::std::uint16_t>(cuda::ptx::mul_mode_hi);
  test_mul<cuda::std::int32_t, cuda::std::int32_t>(cuda::ptx::mul_mode_hi);
  test_mul<cuda::std::uint32_t, cuda::std::uint32_t>(cuda::ptx::mul_mode_hi);
  test_mul<cuda::std::int64_t, cuda::std::int64_t>(cuda::ptx::mul_mode_hi);
  test_mul<cuda::std::uint64_t, cuda::std::uint64_t>(cuda::ptx::mul_mode_hi);

  test_mul<cuda::std::int32_t, cuda::std::int16_t>(cuda::ptx::mul_mode_wide);
  test_mul<cuda::std::uint32_t, cuda::std::uint16_t>(cuda::ptx::mul_mode_wide);
  test_mul<cuda::std::int64_t, cuda::std::int32_t>(cuda::ptx::mul_mode_wide);
  test_mul<cuda::std::uint64_t, cuda::std::uint32_t>(cuda::ptx::mul_mode_wide);
}

int main(int, char**)
{
  return 0;
}
