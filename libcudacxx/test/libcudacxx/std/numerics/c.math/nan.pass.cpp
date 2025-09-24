//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cmath>

#include <cuda/std/cassert>
#include <cuda/std/cmath>

__host__ __device__ constexpr bool test()
{
  assert(cuda::std::isnan(cuda::std::nanf("")));
  assert(cuda::std::isnan(cuda::std::nanf("1")));

  assert(cuda::std::isnan(cuda::std::nan("")));
  assert(cuda::std::isnan(cuda::std::nan("1")));

#if _CCCL_HAS_LONG_DOUBLE()
  assert(cuda::std::isnan(cuda::std::nanl("")));
  assert(cuda::std::isnan(cuda::std::nanl("1")));
#endif // _CCCL_HAS_LONG_DOUBLE()

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
