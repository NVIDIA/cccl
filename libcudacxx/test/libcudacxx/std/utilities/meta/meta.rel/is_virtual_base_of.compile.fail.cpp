//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/cassert>
#include <cuda/std/type_traits>

__host__ __device__ constexpr bool test()
{
#if !defined(_CCCL_BUILTIN_IS_VIRTUAL_BASE_OF)
  static_assert(!cuda::std::is_virtual_base_of<int, int>::value);
  static_assert(!cuda::std::is_virtual_base_of_v<int, int>);
#else
  static_assert(false);
#endif
  return true;
}

int main(int, char**)
{
  static_assert(test());
  return 0;
}
