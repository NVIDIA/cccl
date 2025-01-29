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
  assert(T{cuda::fp16{1.0}} == T{1.0});
  assert(T{cuda::fp32{1.0}} == T{1.0});
  assert(T{cuda::fp64{1.0}} == T{1.0});
  // assert(T{cuda::bf16{1.0}} == T{1.0});
  assert(T{float{1.0}} == T{1.0});
  assert(T{double{1.0}} == T{1.0});
#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
  assert(T{long double{1.0}} == T{1.0});
#endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE

  assert(T{int8_t{1}} == T{1.0});
  assert(T{int16_t{1}} == T{1.0});
  assert(T{int32_t{1}} == T{1.0});
  assert(T{int64_t{1}} == T{1.0});
  assert(T{uint8_t{1}} == T{1.0});
  assert(T{uint16_t{1}} == T{1.0});
  assert(T{uint32_t{1}} == T{1.0});
  assert(T{uint64_t{1}} == T{1.0});

  assert(T{bool{1}} == T{1.0});
  assert(T{char{1}} == T{1.0});

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
