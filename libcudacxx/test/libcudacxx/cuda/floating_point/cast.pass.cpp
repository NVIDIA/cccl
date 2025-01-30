//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++11, c++14

#include <cuda/__floating_point_>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>

#include "test_macros.h"

template <class T>
__host__ __device__ constexpr bool test_type()
{
  assert(static_cast<cuda::fp16>(T{1.0}) == cuda::fp16{1.0});
  assert(static_cast<cuda::fp32>(T{1.0}) == cuda::fp32{1.0});
  assert(static_cast<cuda::fp64>(T{1.0}) == cuda::fp64{1.0});
  // assert(static_cast<cuda::bf16>(T{1.0}) == cuda::bf16{1.0});

  assert(static_cast<float>(T{1.0}) == 1.0f);
  assert(static_cast<double>(T{1.0}) == 1.0);
#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
  assert(static_cast<long double>(T{1.0}) == 1.0L);
#endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE

  assert(static_cast<int8_t>(T{1.0}) == 1);
  assert(static_cast<int16_t>(T{1.0}) == 1);
  assert(static_cast<int32_t>(T{1.0}) == 1);
  assert(static_cast<int64_t>(T{1.0}) == 1);
  assert(static_cast<uint8_t>(T{1.0}) == 1);
  assert(static_cast<uint16_t>(T{1.0}) == 1);
  assert(static_cast<uint32_t>(T{1.0}) == 1);
  assert(static_cast<uint64_t>(T{1.0}) == 1);

  assert(static_cast<bool>(T{1.0}) == true);
  assert(static_cast<char>(T{1.0}) == 1);

  return true;
}

__host__ __device__ constexpr bool test()
{
  test_type<cuda::fp16>();
  test_type<cuda::fp32>();
  test_type<cuda::fp64>();
  // test_type<cuda::bf16>();

  return true;
}

int main(int, char**)
{
  test();
  // static_assert(test());

  return 0;
}
